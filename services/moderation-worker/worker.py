import asyncio
import aio_pika
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
import time

from config import config

# Import inference modules
try:
    from nlp.inference_multitask import MultiTaskModerationInference
    USE_MULTITASK = True
except ImportError:
    from nlp.inference import ModerationInference
    USE_MULTITASK = False

from image.inference_image import ImageModerationInference
# Audio moderation disabled

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Database setup with connection pooling
engine = create_engine(
    config.DATABASE_URL,
    pool_size=20,  # Reduced pool size as we batch updates
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(bind=engine)

# Global publisher exchange
publisher_exchange = None

# Batch processing settings
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.2  # 200ms
job_queue = asyncio.Queue()

# --- Load Models ---

# 1. Text Model
if USE_MULTITASK and config.USE_MULTITASK_MODEL:
    logger.info("Loading Multi-Task PhoBERT model...")
    text_inference_model = MultiTaskModerationInference(
        model_path=config.MODEL_PATH,
        device=config.MODEL_DEVICE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
    logger.info(f"Multi-Task model loaded from {config.MODEL_PATH}")
else:
    logger.info("Loading baseline inference model...")
    text_inference_model = ModerationInference(
        model_path=config.MODEL_PATH,
        device=config.MODEL_DEVICE
    )
    logger.info("Baseline model loaded")

# 2. Image Model (with OCR + Text Moderation)
logger.info("Loading Image Moderation model...")
try:
    image_inference_model = ImageModerationInference(
        model_name=config.IMAGE_MODEL_NAME, 
        device=config.MODEL_DEVICE,
        text_moderator=text_inference_model  # For OCR text moderation
    )
    logger.info("Image model loaded with OCR support.")
except Exception as e:
    logger.error(f"Failed to load image model: {e}")
    image_inference_model = None

# Audio moderation disabled
audio_inference_model = None


async def process_job(message: aio_pika.IncomingMessage):
    """
    Enqueue job for batch processing.
    Do NOT ack here, will be acked after batch processing.
    """
    await job_queue.put(message)


async def process_batch(messages):
    """Process a batch of messages handling mixed types (text, image, audio)"""
    if not messages:
        return

    start_time = datetime.utcnow()
    
    # 1. Parse all messages & Group by type
    valid_messages = []
    job_data_list = []
    
    text_indices = []
    text_inputs = []
    
    image_indices = []
    
    for idx, msg in enumerate(messages):
        try:
            body = json.loads(msg.body.decode('utf-8'))
            job_data_list.append(body)
            valid_messages.append(msg)
            
            # Determine type (default 'text')
            job_type = body.get('type', 'text')
            job_id = body.get('job_id', 'unknown')
            content_preview = str(body.get('text', ''))[:100]
            logger.info(f"[DEBUG] Job {job_id}: type='{job_type}', content_preview='{content_preview}...'")
            
            if job_type == 'text':
                text_indices.append(idx)
                text_inputs.append(body['text'])
            elif job_type == 'image':
                image_indices.append(idx)
            elif job_type == 'audio':
                # Audio moderation disabled - treat as review
                logger.warning(f"Audio moderation disabled - job {job_id} marked for manual review")
                pass
            else:
                # Fallback to text if unknown
                text_indices.append(idx)
                text_inputs.append(body.get('text', ''))

        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            await msg.nack(requeue=False)

    if not valid_messages:
        return

    # Placeholder for all results
    results = [None] * len(valid_messages)
    loop = asyncio.get_running_loop()

    try:
        # --- PROCESS TEXT BATCH ---
        if text_inputs:
            # Run in executor
            text_results = await loop.run_in_executor(
                None, 
                lambda: text_inference_model.batch_predict(text_inputs, batch_size=len(text_inputs))
            )
            # Map back to results
            for i, res in zip(text_indices, text_results):
                results[i] = res

        import base64
        import io
        import tempfile
        
        def decode_base64_file(data_string):
            """Decode base64 string to bytes"""
            if ',' in data_string:
                header, encoded = data_string.split(",", 1)
            else:
                encoded = data_string
            return base64.b64decode(encoded)

        def is_base64(s):
            return isinstance(s, str) and (s.startswith('data:') or len(s) > 200 and ' ' not in s)

        # --- PROCESS IMAGES ---
        if image_indices:
            logger.info(f"[DEBUG] Processing {len(image_indices)} image(s)")
            if image_inference_model:
                for idx in image_indices:
                    content = job_data_list[idx].get('content') or job_data_list[idx].get('text')
                    job_id = job_data_list[idx].get('job_id', 'unknown')
                    logger.info(f"[DEBUG] Image {job_id}: is_base64={is_base64(content)}, content_len={len(content) if content else 0}")
                    
                    try:
                        # Handle Base64
                        if is_base64(content):
                            # ImageInference can load from bytes/stream usually, but let's check implementation
                            # inference_image.py uses PIL.Image.open(path) or requests.get(url)
                            # We can pass BytesIO for PIL
                            image_bytes = decode_base64_file(content)
                            logger.info(f"[DEBUG] Image {job_id}: decoded {len(image_bytes)} bytes")
                            res = await loop.run_in_executor(
                                None,
                                lambda: image_inference_model.predict(io.BytesIO(image_bytes))
                            )
                        else:
                            res = await loop.run_in_executor(
                                None,
                                lambda: image_inference_model.predict(content)
                            )
                        logger.info(f"[DEBUG] Image {job_id} result: {res}")
                        results[idx] = res
                    except Exception as e:
                        logger.error(f"Image processing error for {job_id}: {e}")
                        results[idx] = {'action': 'review', 'reasoning': f'Error: {str(e)}'}
            else:
                for idx in image_indices:
                    results[idx] = {
                        'action': 'review',
                        'confidence': 0.0,
                        'reasoning': 'Image model not available'
                    }

        # Audio moderation disabled - no processing
        
        # --- FINALIZE & UPDATE DB ---
        end_time = datetime.utcnow()
        avg_duration_ms = int((end_time - start_time).total_seconds() * 1000) / len(valid_messages)
        
        db_updates = []
        completion_events = []
        
        for i, (job_data, result) in enumerate(zip(job_data_list, results)):
            if result is None: continue # Should not happen
            
            job_id = job_data['job_id']
            
            # Normalize result format
            # New format logic
            if 'action' in result:
                moderation_result = result['action'] # 'allowed', 'reject', 'review'
                sentiment = result.get('sentiment')
                if not sentiment:
                     sentiment = 'negative' if moderation_result in ['reject'] else 'positive'
                
                confidence = result.get('confidence', 0.0)
                reasoning = result.get('reasoning', '')
                
                if 'labels' in result and result['labels']:
                    reasoning += f" | Labels: {', '.join(result['labels'])}"
            else:
                # Fallback / Old format
                moderation_result = result.get('moderation_result', 'review')
                sentiment = result.get('sentiment', 'neutral')
                confidence = result.get('confidence', 0.0)
                reasoning = result.get('reasoning', '')

            # Prepare DB update
            db_updates.append({
                "job_id": job_id,
                "status": "completed",
                "sentiment": sentiment,
                "moderation_result": moderation_result,
                "confidence_score": confidence,
                "reasoning": reasoning,
                "completed_at": end_time,
                "processing_duration_ms": int(avg_duration_ms)
            })
            
            # Prepare webhook/event data
            event_data = {
                "job_id": job_id,
                "client_id": job_data['client_id'],
                "comment_id": job_data.get('comment_id'),
                "text": job_data.get('text', ''), # original text/url
                "type": job_data.get('type', 'text'),
                "sentiment": sentiment,
                "moderation_result": moderation_result,
                "confidence": confidence,
                "reasoning": reasoning,
                "processing_duration_ms": int(avg_duration_ms),
                "completed_at": end_time.isoformat()
            }
            # Add extra fields if present
            if 'detected_labels' in result:
                event_data['detected_labels'] = result['detected_labels']
            elif 'labels' in result:
                event_data['detected_labels'] = result['labels']
                
            if 'severity_score' in result:
                event_data['severity_score'] = result['severity_score']
            
            if 'transcribed_text' in result:
                event_data['transcribed_text'] = result['transcribed_text']
            
            # Thêm extracted_text từ OCR để debug
            if 'extracted_text' in result:
                event_data['extracted_text'] = result['extracted_text']
                
            completion_events.append(event_data)

        # Bulk Update DB
        db = SessionLocal()
        try:
            db.execute(
                text("""
                    UPDATE jobs 
                    SET status = :status,
                        sentiment = :sentiment,
                        moderation_result = :moderation_result,
                        confidence_score = :confidence_score,
                        reasoning = :reasoning,
                        completed_at = :completed_at,
                        processing_duration_ms = :processing_duration_ms
                    WHERE job_id = :job_id
                """),
                db_updates
            )
            db.commit()
        except Exception as e:
            logger.error(f"DB Bulk Update failed: {e}")
            db.rollback()
        finally:
            db.close()
            
        # Publish Events & Ack
        for i, msg in enumerate(valid_messages):
            # Publish event
            if publisher_exchange:
                try:
                    await publisher_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(completion_events[i]).encode('utf-8'),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                        ),
                        routing_key="moderation.job.completed"
                    )
                except Exception as e:
                    logger.error(f"Failed to publish event: {e}")
            
            # Ack message
            await msg.ack()
            
        logger.info(f"Processed batch of {len(valid_messages)} jobs in {avg_duration_ms*len(valid_messages):.2f}ms")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        # Nack all messages to retry
        for msg in valid_messages:
            await msg.nack(requeue=True)


async def batch_processor_task():
    """Background task to drain queue and process batches"""
    logger.info("Batch processor started")
    while True:
        messages = []
        try:
            # Wait for first message
            msg = await job_queue.get()
            messages.append(msg)
            
            # Collect more messages up to BATCH_SIZE or timeout
            start_wait = time.time()
            while len(messages) < BATCH_SIZE:
                remaining = BATCH_TIMEOUT - (time.time() - start_wait)
                if remaining <= 0:
                    break
                
                try:
                    # Use asyncio.wait_for with timeout
                    msg = await asyncio.wait_for(job_queue.get(), timeout=remaining)
                    messages.append(msg)
                except asyncio.TimeoutError:
                    break
                except Exception:
                    break
            
            # Process the collected batch
            await process_batch(messages)
            
            # Mark tasks as done
            for _ in messages:
                job_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in batch processor: {e}", exc_info=True)
            await asyncio.sleep(1)


async def main():
    """Main worker loop"""
    logger.info("Starting moderation worker (BATCH MODE)...")
    
    # Connect to RabbitMQ
    # Connect to RabbitMQ with retry logic
    connection = None
    retries = 10
    retry_delay = 5
    
    for attempt in range(retries):
        try:
            connection = await aio_pika.connect_robust(
                config.RABBITMQ_URL,
                timeout=30,
                heartbeat=600
            )
            logger.info("Successfully connected to RabbitMQ")
            break
        except Exception as e:
            logger.warning(f"Failed to connect to RabbitMQ (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                logger.error("Max retries reached. Exiting.")
                raise e
            logger.info(f"Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
    channel = await connection.channel()
    
    # Set prefetch count higher to allow buffering
    await channel.set_qos(prefetch_count=config.WORKER_CONCURRENCY)
    
    # Declare exchange
    global publisher_exchange
    publisher_exchange = await channel.declare_exchange(
        "moderation_exchange",
        aio_pika.ExchangeType.DIRECT,
        durable=True
    )
    exchange = publisher_exchange
    
    # Declare queue
    queue = await channel.declare_queue("moderation_jobs", durable=True)
    await queue.bind(exchange, routing_key="moderation.job.new")
    
    logger.info(f"Worker ready. Batch size: {BATCH_SIZE}, Prefetch: {config.WORKER_CONCURRENCY}")
    
    # Start batch processor
    processor = asyncio.create_task(batch_processor_task())
    
    # Start consuming (pushes to queue)
    await queue.consume(process_job)
    
    # Keep running
    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)

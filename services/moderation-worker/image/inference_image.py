from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import requests
import torch
import io
import logging
import numpy as np
import re
import unicodedata

logger = logging.getLogger(__name__)

# ==================== CRITICAL OCR WORDS ====================
# ONLY CHECK FOR CLEARLY TOXIC WORDS - EXACT MATCH ONLY
# Do not use pattern matching or accent-removal to avoid false positives

# CLEARLY TOXIC ABBREVIATIONS - unmistakable
CRITICAL_ABBREVIATIONS = {
    # Viết tắt chửi thề - chỉ match exact
    'dm', 'đm', 'dcm', 'đcm', 
    'vcl', 'vl', 'vkl', 
    'cc', 'clm', 'ctm',
    'dmm', 'đmm', 'dkm', 'đkm', 'vcc', 'cmm',
}

# SEVERE VULGAR WORDS - exact match with Vietnamese diacritics only
CRITICAL_VULGAR_EXACT = {
    # Exact match - có dấu đầy đủ
    'đụ', 'địt', 'lồn', 'cặc', 'buồi', 'đéo', 'cứt',
    'đụ má', 'địt mẹ', 'đéo mẹ',
    
    # English profanity - exact match
    'fuck', 'shit', 'bitch', 'dick', 'cock',
    'fucking', 'motherfucker',
}

# WORDS REQUIRING CONTEXT - DO NOT auto-reject, let text_moderator handle
# These words are highly prone to false positives:
# - 'ngu' (nguồn, người, ngủ, nguyen...)
# - 'dốt' (đột, đốt...)  
# - 'chó' (cho, chọn...)
# - 'lợn' (lòng, lớn...)
# - 'gay' (gay gắt, đồng tính gay is not hate speech)
# - 'ass' (class, pass, bass...)

# EMPTY - không dùng pattern matching nữa vì gây quá nhiều false positive
OCR_TOXIC_PATTERNS = []

# ==================== OCR CONFIGURATION ====================
# Use OCR.space API as primary OCR (free, supports Vietnamese)
# Falls back to EasyOCR if OCR.space unavailable

import os

# Check which OCR service to use
USE_OCRSPACE_OCR = os.getenv('USE_OCRSPACE_OCR', 'true').lower() == 'true'
OCRSPACE_API_KEY = os.getenv('OCRSPACE_API_KEY', 'helloworld')  # Free default key

# OCR.space Vision OCR instance
_ocrspace_ocr = None

def get_ocrspace_ocr():
    """Get OCR.space Vision OCR instance"""
    global _ocrspace_ocr
    if _ocrspace_ocr is None and USE_OCRSPACE_OCR:
        try:
            from .ocrspace_vision import OCRSpaceVision
            logger.info("Initializing OCR.space Vision OCR...")
            _ocrspace_ocr = OCRSpaceVision(api_key=OCRSPACE_API_KEY, language="auto")
            logger.info("OCR.space Vision OCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR.space OCR: {e}")
            _ocrspace_ocr = False
    return _ocrspace_ocr if _ocrspace_ocr else None

# EasyOCR Reader - used as fallback
_ocr_reader = None

def get_easyocr_reader():
    """Lazy load EasyOCR reader as fallback"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR as fallback for text extraction...")
            _ocr_reader = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            _ocr_reader = False  # Mark as failed, don't retry
    return _ocr_reader if _ocr_reader else None


# Keep legacy function name for compatibility
def get_ocr_reader():
    """Legacy function - returns EasyOCR reader"""
    return get_easyocr_reader()


def normalize_text(text):
    """Normalize text for matching - lowercase, remove accents for comparison"""
    if not text:
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_vietnamese_accents(text):
    """Remove Vietnamese accents for fuzzy matching"""
    # Map Vietnamese characters to ASCII equivalents
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
    }
    result = []
    for char in text.lower():
        result.append(vietnamese_map.get(char, char))
    return ''.join(result)


def check_critical_ocr_text(text):
    """
    SIMPLIFIED Critical OCR text check - ONLY exact match with clearly vulgar words
    
    Strategy:
    1. CHỈ kiểm tra viết tắt chửi thề (dm, vcl, cc...) - EXACT MATCH
    2. CHỈ kiểm tra từ thô tục CÓ DẤU ĐẦY ĐỦ (đụ, địt, lồn...) - EXACT MATCH
    3. KHÔNG dùng pattern matching hoặc accent-removal
    4. Các trường hợp không rõ ràng -> để text_moderator ML xử lý
    
    Returns:
        dict or None: Violation result if found, None otherwise
    """
    if not text or len(text.strip()) < 2:
        return None
    
    text_lower = text.lower().strip()
    words_in_text = text_lower.split()
    
    logger.info(f"[OCR-CRITICAL-V2] Checking text: '{text}' -> words: {words_in_text}")
    
    detected_words = []
    
    # ===== CHECK 1: Viết tắt chửi thề - EXACT MATCH =====
    # Các từ này không thể nhầm lẫn trong ngữ cảnh khác
    for word in words_in_text:
        word_clean = word.strip('.,!?;:"\x27()[]{}')  # Strip punctuation
        if word_clean in CRITICAL_ABBREVIATIONS:
            detected_words.append(word_clean)
            logger.info(f"[OCR-CRITICAL-V2] FOUND abbreviation: '{word_clean}'")
    
    # ===== CHECK 2: Từ thô tục có dấu - EXACT MATCH =====
    # Chỉ match khi từ CHÍNH XÁC có dấu đầy đủ
    for word in words_in_text:
        word_clean = word.strip('.,!?;:"\'()[]{}')  # Strip punctuation
        if word_clean in CRITICAL_VULGAR_EXACT:
            detected_words.append(word_clean)
            logger.info(f"[OCR-CRITICAL-V2] FOUND vulgar word: '{word_clean}'")
    
    # ===== CHECK 3: Cụm từ thô tục - EXACT MATCH =====
    # Check các cụm từ như "đụ má", "địt mẹ"
    for phrase in CRITICAL_VULGAR_EXACT:
        if ' ' in phrase and phrase in text_lower:
            detected_words.append(phrase)
            logger.info(f"[OCR-CRITICAL-V2] FOUND vulgar phrase: '{phrase}'")
    
    # KHÔNG kiểm tra OCR_TOXIC_PATTERNS vì danh sách đã rỗng
    # Pattern matching gây quá nhiều false positive nên đã loại bỏ
    
    if detected_words:
        unique_words = list(set(detected_words))
        logger.info(f"[OCR-CRITICAL-V2] VIOLATION DETECTED: {unique_words}")
        return {
            'is_violation': True,
            'detected_words': unique_words,
            'action': 'reject',
            'confidence': 0.98,  # Cao vì exact match
            'reasoning': f"Bypass filter detected: {', '.join(unique_words)} (obfuscation)"
        }
    
    logger.info(f"[OCR-CRITICAL-V2] No critical words found in OCR text - will use ML text_moderator")
    return None


class ImageModerationInference:
    def __init__(self, model_name="Falconsai/nsfw_image_detection", device="cpu", text_moderator=None):
        """
        Initialize image moderation with NSFW detection + Enhanced OCR text extraction
        
        Args:
            model_name: HuggingFace model for NSFW detection
            device: 'cpu' or 'cuda'
            text_moderator: Optional text moderation model for checking extracted text
        """
        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.text_moderator = text_moderator
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"Loading image moderation model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
            logger.info("Image moderation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image moderation model: {e}")
            raise
    
    def set_text_moderator(self, text_moderator):
        """Set text moderator after initialization"""
        self.text_moderator = text_moderator
        logger.info("Text moderator attached to ImageModerationInference")

    def preprocess_image_for_ocr(self, image):
        """
        Preprocess image for better OCR results
        Returns list of preprocessed image variants
        """
        variants = []
        
        # Original
        variants.append(("original", np.array(image)))
        
        try:
            # 1. Grayscale with contrast enhancement
            gray = ImageOps.grayscale(image)
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(2.0)
            variants.append(("high_contrast", np.array(high_contrast.convert("RGB"))))
            
            # 2. Sharpen
            sharp = image.filter(ImageFilter.SHARPEN)
            variants.append(("sharp", np.array(sharp)))
            
            # 3. Resize (upscale small images)
            width, height = image.size
            if width < 500 or height < 500:
                scale = max(500 / width, 500 / height)
                new_size = (int(width * scale), int(height * scale))
                upscaled = image.resize(new_size, Image.LANCZOS)
                variants.append(("upscaled", np.array(upscaled)))
            
            # 4. Invert colors (for dark backgrounds)
            inverted = ImageOps.invert(image.convert("RGB"))
            variants.append(("inverted", np.array(inverted)))
            
            # 5. Binary threshold (for text detection)
            gray_np = np.array(ImageOps.grayscale(image))
            _, binary = np.array([gray_np > 127], dtype=np.uint8)[0] * 255, None
            # Skip binary if it fails
            
        except Exception as e:
            logger.warning(f"Image preprocessing variant failed: {e}")
        
        return variants

    def extract_text_from_image(self, image):
        """
        Enhanced text extraction from image using OpenRouter Vision API
        Falls back to EasyOCR if OpenRouter is unavailable
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text or empty string
        """
        
        # ===== TRY OCR.SPACE API FIRST (Free, supports Vietnamese) =====
        ocrspace_ocr = get_ocrspace_ocr()
        if ocrspace_ocr:
            try:
                logger.info("[OCR] Using OCR.space API for text extraction...")
                extracted_text = ocrspace_ocr.extract_text(image)
                
                if extracted_text:
                    logger.info(f"[OCR-Space] Extracted text: '{extracted_text}'")
                    return extracted_text
                else:
                    logger.info("[OCR-Space] No text found in image")
                    return ""
                    
            except Exception as e:
                logger.warning(f"[OCR-Space] Failed: {e}, falling back to EasyOCR")
        
        # ===== FALLBACK TO EASYOCR =====
        logger.info("[OCR] Falling back to EasyOCR...")
        ocr_reader = get_easyocr_reader()
        if not ocr_reader:
            logger.warning("No OCR method available, skipping text extraction")
            return ""
        
        all_extracted_texts = []
        
        try:
            # Get preprocessed variants
            variants = self.preprocess_image_for_ocr(image)
            
            for variant_name, image_np in variants:
                try:
                    # Run OCR with low confidence threshold
                    results = ocr_reader.readtext(image_np, detail=1, paragraph=False)
                    
                    if not results:
                        continue
                    
                    logger.info(f"[OCR-EasyOCR-{variant_name}] Found {len(results)} text regions")
                    
                    for (bbox, text, conf) in results:
                        logger.info(f"[OCR-EasyOCR-{variant_name}] Text: '{text}' (conf: {conf:.2f})")
                        
                        # Very low threshold - we want to catch everything
                        if conf > 0.1:  # Reduced from 0.3 to 0.1
                            cleaned_text = text.strip()
                            if cleaned_text and len(cleaned_text) >= 2:  # At least 2 chars
                                all_extracted_texts.append(cleaned_text)
                
                except Exception as e:
                    logger.warning(f"EasyOCR on {variant_name} variant failed: {e}")
                    continue
            
            # Combine and deduplicate
            if all_extracted_texts:
                # Remove duplicates while preserving order
                seen = set()
                unique_texts = []
                for text in all_extracted_texts:
                    text_lower = text.lower()
                    if text_lower not in seen:
                        seen.add(text_lower)
                        unique_texts.append(text)
                
                full_text = " ".join(unique_texts)
                logger.info(f"[OCR-EasyOCR-FINAL] Extracted text from image: '{full_text}'")
                return full_text
            
            logger.info("[OCR-EasyOCR-FINAL] No text extracted from image")
            return ""
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""

    def predict(self, image_url_or_path_or_bytes):
        try:
            # 1. Load image
            if isinstance(image_url_or_path_or_bytes, (str, bytes, io.BytesIO)):
                if isinstance(image_url_or_path_or_bytes, str):
                    if image_url_or_path_or_bytes.startswith('http'):
                        response = requests.get(image_url_or_path_or_bytes, stream=True, timeout=10)
                        response.raise_for_status()
                        image = Image.open(response.raw)
                    else:
                        image = Image.open(image_url_or_path_or_bytes)
                else:
                    # Bytes or BytesIO
                    if isinstance(image_url_or_path_or_bytes, bytes):
                        image = Image.open(io.BytesIO(image_url_or_path_or_bytes))
                    else:
                        image = Image.open(image_url_or_path_or_bytes)
            else:
                raise ValueError("Invalid image input type")
            
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            logger.info(f"[IMAGE] Processing image: {image.size[0]}x{image.size[1]}")
            
            # ===== STEP 1: NSFW Detection =====
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=1)
                
            top_prob, top_label_idx = probs.max(dim=1)
            label = self.model.config.id2label[top_label_idx.item()]
            nsfw_confidence = top_prob.item()
            
            is_nsfw = label.lower() == 'nsfw'
            
            logger.info(f"[NSFW] Detection result: {label} ({nsfw_confidence:.2%})")
            
            # ===== STEP 2: Enhanced OCR Text Extraction =====
            extracted_text = self.extract_text_from_image(image)
            
            # ===== STEP 3: Text Moderation using main text moderator =====
            # CHANGED: No longer using separate critical word check
            # All OCR text goes through the main text_moderator which uses:
            # - variant_detector.py (with safe_contexts)
            # - toxic_words.py
            # - ML model
            # This ensures consistent moderation logic across text and image
            text_moderation_result = None
            
            if extracted_text and self.text_moderator:
                logger.info(f"[OCR] Running text moderation on extracted text: '{extracted_text}'")
                try:
                    text_moderation_result = self.text_moderator.predict(extracted_text)
                    logger.info(f"[OCR] Text moderation result: {text_moderation_result}")
                except Exception as e:
                    logger.error(f"Text moderation on OCR text failed: {e}")
            
            # ===== STEP 4: Combine Results =====
            # Priority: NSFW > Text moderation result
            
            if is_nsfw:
                # NSFW image - reject immediately
                return {
                    'moderation_result': 'reject',
                    'sentiment': 'negative',
                    'label': label,
                    'confidence': nsfw_confidence,
                    'reasoning': f"NSFW image detected: {label} ({nsfw_confidence:.2%})",
                    'extracted_text': extracted_text,
                    'text_moderation': text_moderation_result
                }
            
            if text_moderation_result:
                text_action = text_moderation_result.get('moderation_result') or text_moderation_result.get('action', 'allowed')
                text_reasoning = text_moderation_result.get('reasoning', '')
                text_confidence = text_moderation_result.get('confidence', 0.0)
                detected_labels = text_moderation_result.get('labels', [])
                
                logger.info(f"[TEXT-MOD] action={text_action}, labels={detected_labels}, reasoning={text_reasoning}")
                
                if text_action == 'reject':
                    # Text in image violates policy
                    return {
                        'moderation_result': 'reject',
                        'sentiment': 'negative',
                        'label': 'text_violation',
                        'confidence': text_confidence,
                        'reasoning': f"Policy violation text detected in image: '{extracted_text}' - {text_reasoning}",
                        'extracted_text': extracted_text,
                        'detected_labels': detected_labels,
                        'image_label': label,
                        'image_confidence': nsfw_confidence
                    }
                elif text_action == 'review':
                    return {
                        'moderation_result': 'review',
                        'sentiment': 'neutral',
                        'label': 'text_review',
                        'confidence': text_confidence,
                        'reasoning': f"Text in image needs review: '{extracted_text}' - {text_reasoning}",
                        'extracted_text': extracted_text,
                        'detected_labels': detected_labels,
                        'image_label': label,
                        'image_confidence': nsfw_confidence
                    }
            
            # Image is normal and text (if any) is acceptable
            reasoning_parts = [f"Image detected: {label} ({nsfw_confidence:.2%})"]
            # Always show OCR results for debugging
            if extracted_text:
                reasoning_parts.append(f"OCR: '{extracted_text}' - No violation")
            else:
                reasoning_parts.append("OCR: No text found in image")
            
            return {
                'moderation_result': 'allowed',
                'sentiment': 'positive',
                'label': label,
                'confidence': nsfw_confidence,
                'reasoning': " | ".join(reasoning_parts),
                'extracted_text': extracted_text if extracted_text else ""
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {
                'moderation_result': 'review',
                'sentiment': 'neutral',
                'label': 'error',
                'confidence': 0.0,
                'reasoning': f"Image processing error: {str(e)}"
            }

"""
OpenRouter Vision API for OCR text extraction from images
Uses free models like Qwen2.5-VL for accurate Vietnamese OCR
"""

import os
import base64
import logging
import requests
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# OpenRouter API Configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free Vision Models - verified to support image input
FREE_VISION_MODELS = [
    "google/gemma-3-27b-it:free",             # Gemma 3 - supports vision
    "google/gemma-3-12b-it:free",             # Gemma 3 smaller
    "qwen/qwen-2.5-vl-7b-instruct:free",      # Qwen VL
    "nvidia/nemotron-nano-2-vl:free",         # NVIDIA Nemotron for OCR
]


class OpenRouterVisionOCR:
    """
    OCR using OpenRouter's free Vision models
    Much more accurate than EasyOCR for complex images
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenRouter Vision OCR
        
        Args:
            api_key: OpenRouter API key (or from env OPENROUTER_API_KEY)
            model: Specific model to use (or uses best free model)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or FREE_VISION_MODELS[0]
        self.current_model_index = 0
        
        if not self.api_key:
            logger.warning("OpenRouter API key not set! OCR will not work.")
        else:
            logger.info(f"OpenRouter Vision OCR initialized with model: {self.model}")
    
    def _image_to_base64(self, image) -> str:
        """Convert PIL Image to base64 string"""
        if isinstance(image, str):
            # File path
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(image, BytesIO):
            image.seek(0)
            return base64.b64encode(image.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _switch_to_next_model(self):
        """Switch to next available model if current one fails"""
        self.current_model_index += 1
        if self.current_model_index < len(FREE_VISION_MODELS):
            self.model = FREE_VISION_MODELS[self.current_model_index]
            logger.info(f"Switching to fallback model: {self.model}")
            return True
        return False
    
    def extract_text(self, image, max_retries: int = 2) -> str:
        """
        Extract text from image using OpenRouter Vision API
        
        Args:
            image: PIL Image, file path, bytes, or BytesIO
            max_retries: Number of retries with different models
            
        Returns:
            str: Extracted text or empty string if failed
        """
        if not self.api_key:
            logger.error("Cannot extract text: OpenRouter API key not set")
            return ""
        
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://vietcms.local",  # Required by OpenRouter
                "X-Title": "VietCMS OCR"
            }
            
            # OCR-focused prompt in Vietnamese
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract ALL text contained in this image. 
Only return the extracted text, do not add any explanation or description.
If no text is found, return "NO TEXT FOUND".
Maintain the original format of the text (uppercase/lowercase, punctuation)."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for accuracy
            }
            
            # Make API request
            logger.info(f"[OpenRouter-OCR] Sending request to {self.model}...")
            
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Clean up the response
                extracted_text = extracted_text.strip()
                
                # Check for "no text" responses
                no_text_indicators = [
                    "KHÔNG CÓ VĂN BẢN",
                    "NO TEXT FOUND", 
                    "no text",
                    "không có văn bản",
                    "không tìm thấy",
                    "empty",
                    "trống"
                ]
                
                for indicator in no_text_indicators:
                    if indicator.lower() in extracted_text.lower():
                        logger.info("[OpenRouter-OCR] No text found in image")
                        return ""
                
                logger.info(f"[OpenRouter-OCR] Extracted text: '{extracted_text}'")
                return extracted_text
                
            elif response.status_code == 429:
                # Rate limited - try next model
                logger.warning(f"[OpenRouter-OCR] Rate limited on {self.model}")
                if max_retries > 0 and self._switch_to_next_model():
                    return self.extract_text(image, max_retries - 1)
                return ""
                
            elif response.status_code == 402:
                # Payment required (credits exhausted)
                logger.error("[OpenRouter-OCR] Credits exhausted! Please add credits to OpenRouter account.")
                return ""
                
            else:
                error_msg = response.json().get("error", {}).get("message", response.text)
                logger.error(f"[OpenRouter-OCR] API error {response.status_code}: {error_msg}")
                
                # Try next model on error
                if max_retries > 0 and self._switch_to_next_model():
                    return self.extract_text(image, max_retries - 1)
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("[OpenRouter-OCR] Request timed out")
            if max_retries > 0 and self._switch_to_next_model():
                return self.extract_text(image, max_retries - 1)
            return ""
            
        except Exception as e:
            logger.error(f"[OpenRouter-OCR] Error: {e}", exc_info=True)
            return ""


# Global instance for lazy loading
_openrouter_ocr = None


def get_openrouter_ocr(api_key: str = None) -> OpenRouterVisionOCR:
    """
    Get or create OpenRouter OCR instance (singleton pattern)
    
    Args:
        api_key: Optional API key to use
        
    Returns:
        OpenRouterVisionOCR instance
    """
    global _openrouter_ocr
    
    if _openrouter_ocr is None:
        _openrouter_ocr = OpenRouterVisionOCR(api_key=api_key)
    
    return _openrouter_ocr


def extract_text_with_openrouter(image, api_key: str = None) -> str:
    """
    Convenience function to extract text from image
    
    Args:
        image: PIL Image, file path, bytes, or BytesIO
        api_key: Optional API key
        
    Returns:
        str: Extracted text or empty string
    """
    ocr = get_openrouter_ocr(api_key)
    return ocr.extract_text(image)

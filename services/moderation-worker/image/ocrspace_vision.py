"""
OCR.space API for OCR text extraction from images
Free API that supports Vietnamese language
https://ocr.space/OCRAPI
"""

import os
import base64
import logging
import requests
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# OCR.space API Configuration
OCRSPACE_API_URL = "https://api.ocr.space/parse/image"

# Free API key - can get your own at https://ocr.space/ocrapi
# The free tier allows 500 requests/day
DEFAULT_API_KEY = "helloworld"  # Default free key for testing


class OCRSpaceVision:
    """
    OCR using OCR.space free API
    Supports Vietnamese (vnm) and many other languages
    """
    
    def __init__(self, api_key: str = None, language: str = "auto"):
        """
        Initialize OCR.space Vision OCR
        
        Args:
            api_key: OCR.space API key (or from env OCRSPACE_API_KEY)
            language: OCR language code (auto=auto-detect, eng=English)
                      Note: For Vietnamese, use 'auto' with Engine 2
        """
        self.api_key = api_key or os.getenv("OCRSPACE_API_KEY", DEFAULT_API_KEY)
        self.language = language
        
        logger.info(f"OCR.space Vision initialized with language: {self.language}")
    
    def _image_to_base64(self, image) -> str:
        """Convert PIL Image to base64 string with data URI prefix"""
        if isinstance(image, str):
            # File path
            with open(image, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{data}"
        elif isinstance(image, bytes):
            data = base64.b64encode(image).decode("utf-8")
            return f"data:image/png;base64,{data}"
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="PNG", quality=85)
            data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        elif isinstance(image, BytesIO):
            image.seek(0)
            data = base64.b64encode(image.read()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def extract_text(self, image, max_retries: int = 2) -> str:
        """
        Extract text from image using OCR.space API
        
        Args:
            image: PIL Image, file path, bytes, or BytesIO
            max_retries: Number of retries on failure
            
        Returns:
            str: Extracted text or empty string if failed
        """
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Prepare the request
            payload = {
                "apikey": self.api_key,
                "base64Image": image_base64,
                "language": self.language,
                "isOverlayRequired": False,
                "detectOrientation": True,
                "scale": True,
                "OCREngine": 2,  # Engine 2 is better for accuracy
            }
            
            # Make API request
            logger.info(f"[OCRSpace] Sending request with language={self.language}...")
            
            response = requests.post(
                OCRSPACE_API_URL,
                data=payload,
                timeout=30
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                
                # Check for errors in response
                if result.get("IsErroredOnProcessing", False):
                    error_msg = result.get("ErrorMessage", ["Unknown error"])
                    logger.error(f"[OCRSpace] Processing error: {error_msg}")
                    return ""
                
                # Extract text from parsed results
                parsed_results = result.get("ParsedResults", [])
                if not parsed_results:
                    logger.info("[OCRSpace] No text found in image")
                    return ""
                
                # Combine text from all parsed results
                extracted_texts = []
                for parsed in parsed_results:
                    text = parsed.get("ParsedText", "").strip()
                    if text:
                        extracted_texts.append(text)
                
                if not extracted_texts:
                    logger.info("[OCRSpace] No text found in image")
                    return ""
                
                full_text = "\n".join(extracted_texts)
                logger.info(f"[OCRSpace] Extracted text: '{full_text[:100]}...'")
                return full_text
                
            else:
                logger.error(f"[OCRSpace] API error {response.status_code}: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("[OCRSpace] Request timed out")
            if max_retries > 0:
                logger.info(f"[OCRSpace] Retrying... ({max_retries} retries left)")
                return self.extract_text(image, max_retries - 1)
            return ""
            
        except Exception as e:
            logger.error(f"[OCRSpace] Error: {e}", exc_info=True)
            return ""


# Global instance for lazy loading
_ocrspace_ocr = None


def get_ocrspace_ocr(api_key: str = None, language: str = "auto") -> OCRSpaceVision:
    """
    Get or create OCR.space OCR instance (singleton pattern)
    
    Args:
        api_key: Optional API key to use
        language: OCR language code
        
    Returns:
        OCRSpaceVision instance
    """
    global _ocrspace_ocr
    
    if _ocrspace_ocr is None:
        _ocrspace_ocr = OCRSpaceVision(api_key=api_key, language=language)
    
    return _ocrspace_ocr


def extract_text_with_ocrspace(image, api_key: str = None, language: str = "auto") -> str:
    """
    Convenience function to extract text from image
    
    Args:
        image: PIL Image, file path, bytes, or BytesIO
        api_key: Optional API key
        language: OCR language code (vie=Vietnamese)
        
    Returns:
        str: Extracted text or empty string
    """
    ocr = get_ocrspace_ocr(api_key, language)
    return ocr.extract_text(image)

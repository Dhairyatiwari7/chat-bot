"""
Fallback OCR System for RAG Narok
Used when Tesseract OCR is not available
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class FallbackOCR:
    """Fallback OCR system when Tesseract is not available"""
    
    def __init__(self):
        self.available = False
        self.setup_available_methods()
    
    def setup_available_methods(self):
        """Setup available OCR methods"""
        available_methods = []
        
        # Check for basic image processing
        try:
            import cv2
            available_methods.append("opencv")
        except ImportError:
            pass
        
        try:
            from PIL import Image
            available_methods.append("pillow")
        except ImportError:
            pass
        
        if available_methods:
            self.available = True
            logger.info(f"Fallback OCR methods available: {available_methods}")
        else:
            logger.warning("No fallback OCR methods available")
    
    def extract_text_basic(self, image_path: str) -> Dict:
        """Basic text extraction using available methods"""
        try:
            # For now, return a placeholder with image analysis
            import os
            from PIL import Image
            
            # Get basic image information
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
            
            return {
                "extracted_text": f"Image analysis: {width}x{height} {mode} {format_name} format. OCR not available - please install Tesseract for text extraction.",
                "confidence": 0,
                "method": "fallback_basic",
                "image_info": {
                    "dimensions": f"{width}x{height}",
                    "mode": mode,
                    "format": format_name
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback OCR error: {e}")
            return {
                "extracted_text": f"Error analyzing image: {str(e)}",
                "confidence": 0,
                "method": "fallback_error"
            }

# Initialize fallback OCR
fallback_ocr = FallbackOCR()

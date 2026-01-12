"""
Title OCR Service
Extracts Pokemon title/name using Tesseract OCR
"""

from ocr_service import OCRService
import os


class TitleOCRService(OCRService):
    """Service for extracting Pokemon title/name from images"""
    
    def extract_title(self, image_path: str) -> str:
        """
        Extract Pokemon title/name from image
        
        Args:
            image_path: Path to Pokemon image
        
        Returns:
            Extracted Pokemon name/title
        """
        try:
            # Preprocess for better OCR
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is None:
                return None
            
            # Use custom config for title extraction (single line)
            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
            
            text = self.extract_text(image_path, lang="eng")
            
            if text:
                # Clean up the extracted text
                title = text.strip().upper()
                return title
            
            return None
        except Exception as e:
            print(f"Error extracting title: {str(e)}")
            return None
    
    def extract_title_from_region(self, image_path: str, bbox: tuple) -> str:
        """
        Extract title from a specific region of the image
        
        Args:
            image_path: Path to Pokemon image
            bbox: Bounding box (x, y, width, height) for title region
        
        Returns:
            Extracted Pokemon name/title
        """
        try:
            # Crop the region containing the title
            cropped = self.crop_region(image_path, bbox)
            if cropped is None:
                return None
            
            # Save cropped image temporarily
            temp_path = "temp_title_crop.png"
            import cv2
            cv2.imwrite(temp_path, cropped)
            
            # Extract text with single line config
            config = "--psm 7"
            text = self.extract_text_with_config(temp_path, config=config)
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                title = text.strip().upper()
                return title
            
            return None
        except Exception as e:
            print(f"Error extracting title from region: {str(e)}")
            return None

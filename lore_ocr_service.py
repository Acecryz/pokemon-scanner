"""
Lore OCR Service
Extracts Pokemon lore/description text using Tesseract OCR
"""

from name_ocr_service import OCRService
import os


class LoreOCRService(OCRService):
    """Service for extracting Pokemon lore/description from images"""
    
    def extract_lore(self, image_path: str) -> str:
        """
        Extract Pokemon lore/description from image
        
        Args:
            image_path: Path to Pokemon image
        
        Returns:
            Extracted lore text
        """
        try:
            # Preprocess for better OCR
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is None:
                return None
            
            # Extract text with multi-line config
            config = "--psm 6"
            text = self.extract_text_with_config(image_path, config=config)
            
            if text:
                # Clean up the extracted text
                lore = text.strip()
                return lore
            
            return None
        except Exception as e:
            print(f"Error extracting lore: {str(e)}")
            return None
    
    def extract_lore_from_region(self, image_path: str, bbox: tuple) -> str:
        """
        Extract lore from a specific region of the image
        
        Args:
            image_path: Path to Pokemon image
            bbox: Bounding box (x, y, width, height) for lore region
        
        Returns:
            Extracted lore text
        """
        try:
            # Crop the region containing the lore
            cropped = self.crop_region(image_path, bbox)
            if cropped is None:
                return None
            
            # Save cropped image temporarily
            temp_path = "temp_lore_crop.png"
            import cv2
            cv2.imwrite(temp_path, cropped)
            
            # Extract text with multi-line config for paragraph text
            config = "--psm 6"
            text = self.extract_text_with_config(temp_path, config=config)
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                lore = text.strip()
                return lore
            
            return None
        except Exception as e:
            print(f"Error extracting lore from region: {str(e)}")
            return None

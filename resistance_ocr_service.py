"""
Resistance OCR Service
Extracts Pokemon resistance data and saves processed image using Tesseract OCR
"""

from ocr_service import OCRService
from pathlib import Path
import os


class ResistanceOCRService(OCRService):
    """Service for extracting Pokemon resistance information from images"""
    
    def __init__(self, tesseract_path=None):
        """Initialize Resistance OCR Service"""
        super().__init__(tesseract_path)
        self.output_dir = Path("resistances")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_resistance(self, image_path: str) -> str:
        """
        Extract resistance text from image
        
        Args:
            image_path: Path to Pokemon image
        
        Returns:
            Extracted resistance text
        """
        try:
            # Preprocess for better OCR
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is None:
                return None
            
            # Extract text
            text = self.extract_text(image_path, lang="eng")
            
            if text:
                resistance = text.strip()
                return resistance
            
            return None
        except Exception as e:
            print(f"Error extracting resistance: {str(e)}")
            return None
    
    def extract_and_save_resistance_image(self, image_path: str, record_id: str, 
                                         bbox: tuple = None) -> str:
        """
        Extract resistance region and save as separate image
        
        Args:
            image_path: Path to Pokemon image
            record_id: MongoDB record ID
            bbox: Optional bounding box (x, y, width, height) for resistance region
        
        Returns:
            Path to saved resistance image
        """
        try:
            output_filename = f"{record_id}.png"
            output_path = self.output_dir / output_filename
            
            if bbox:
                # Crop specific region
                self.crop_region(image_path, bbox, output_path=str(output_path))
            else:
                # Copy/process entire image
                import cv2
                image = cv2.imread(image_path)
                if image is not None:
                    cv2.imwrite(str(output_path), image)
            
            return str(output_path).replace("\\", "/")
        except Exception as e:
            print(f"Error saving resistance image: {str(e)}")
            return None
    
    def extract_resistance_from_region(self, image_path: str, bbox: tuple) -> str:
        """
        Extract resistance from a specific region of the image
        
        Args:
            image_path: Path to Pokemon image
            bbox: Bounding box (x, y, width, height) for resistance region
        
        Returns:
            Extracted resistance text
        """
        try:
            # Crop the region
            cropped = self.crop_region(image_path, bbox)
            if cropped is None:
                return None
            
            # Save cropped image temporarily
            temp_path = "temp_resistance_crop.png"
            import cv2
            cv2.imwrite(temp_path, cropped)
            
            # Extract text
            text = self.extract_text(temp_path, lang="eng")
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                resistance = text.strip()
                return resistance
            
            return None
        except Exception as e:
            print(f"Error extracting resistance from region: {str(e)}")
            return None

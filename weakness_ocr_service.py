"""
Weakness OCR Service
Extracts Pokemon weakness data and saves processed image using Tesseract OCR
"""

from ocr_service import OCRService
from pathlib import Path
import os


class WeaknessOCRService(OCRService):
    """Service for extracting Pokemon weakness information from images"""
    
    def __init__(self, tesseract_path=None):
        """Initialize Weakness OCR Service"""
        super().__init__(tesseract_path)
        self.output_dir = Path("weaknesses")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_weakness(self, image_path: str) -> str:
        """
        Extract weakness text from image
        
        Args:
            image_path: Path to Pokemon image
        
        Returns:
            Extracted weakness text
        """
        try:
            # Preprocess for better OCR
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is None:
                return None
            
            # Extract text
            text = self.extract_text(image_path, lang="eng")
            
            if text:
                weakness = text.strip()
                return weakness
            
            return None
        except Exception as e:
            print(f"Error extracting weakness: {str(e)}")
            return None
    
    def extract_and_save_weakness_image(self, image_path: str, record_id: str, 
                                       bbox: tuple = None) -> str:
        """
        Extract weakness region and save as separate image
        
        Args:
            image_path: Path to Pokemon image
            record_id: MongoDB record ID
            bbox: Optional bounding box (x, y, width, height) for weakness region
        
        Returns:
            Path to saved weakness image
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
            print(f"Error saving weakness image: {str(e)}")
            return None
    
    def extract_weakness_from_region(self, image_path: str, bbox: tuple) -> str:
        """
        Extract weakness from a specific region of the image
        
        Args:
            image_path: Path to Pokemon image
            bbox: Bounding box (x, y, width, height) for weakness region
        
        Returns:
            Extracted weakness text
        """
        try:
            # Crop the region
            cropped = self.crop_region(image_path, bbox)
            if cropped is None:
                return None
            
            # Save cropped image temporarily
            temp_path = "temp_weakness_crop.png"
            import cv2
            cv2.imwrite(temp_path, cropped)
            
            # Extract text
            text = self.extract_text(temp_path, lang="eng")
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                weakness = text.strip()
                return weakness
            
            return None
        except Exception as e:
            print(f"Error extracting weakness from region: {str(e)}")
            return None

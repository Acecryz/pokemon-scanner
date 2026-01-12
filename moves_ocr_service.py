"""
Moves OCR Service
Extracts Pokemon moves data and saves processed image using Tesseract OCR
"""

from name_ocr_service import OCRService
from pathlib import Path
import os


class MovesOCRService(OCRService):
    """Service for extracting Pokemon moves information from images"""
    
    def __init__(self, tesseract_path=None):
        """Initialize Moves OCR Service"""
        super().__init__(tesseract_path)
        self.output_dir = Path("moves")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_moves(self, image_path: str) -> list:
        """
        Extract moves list from image
        
        Args:
            image_path: Path to Pokemon image
        
        Returns:
            List of extracted moves
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
                # Split by newlines and filter empty lines
                moves = [move.strip() for move in text.split('\n') if move.strip()]
                return moves
            
            return None
        except Exception as e:
            print(f"Error extracting moves: {str(e)}")
            return None
    
    def extract_and_save_moves_image(self, image_path: str, record_id: str, 
                                    bbox: tuple = None) -> str:
        """
        Extract moves region and save as separate image
        
        Args:
            image_path: Path to Pokemon image
            record_id: MongoDB record ID
            bbox: Optional bounding box (x, y, width, height) for moves region
        
        Returns:
            Path to saved moves image
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
            print(f"Error saving moves image: {str(e)}")
            return None
    
    def extract_moves_from_region(self, image_path: str, bbox: tuple) -> list:
        """
        Extract moves from a specific region of the image
        
        Args:
            image_path: Path to Pokemon image
            bbox: Bounding box (x, y, width, height) for moves region
        
        Returns:
            List of extracted moves
        """
        try:
            # Crop the region
            cropped = self.crop_region(image_path, bbox)
            if cropped is None:
                return None
            
            # Save cropped image temporarily
            temp_path = "temp_moves_crop.png"
            import cv2
            cv2.imwrite(temp_path, cropped)
            
            # Extract text
            config = "--psm 6"
            text = self.extract_text_with_config(temp_path, config=config)
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                moves = [move.strip() for move in text.split('\n') if move.strip()]
                return moves
            
            return None
        except Exception as e:
            print(f"Error extracting moves from region: {str(e)}")
            return None

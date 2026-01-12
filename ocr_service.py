"""
Base OCR Service Module
Provides Tesseract OCR functionality for all Pokemon Scanner services
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import pytesseract


class OCRService:
    """Base class for OCR services using Tesseract"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize OCR Service
        
        Args:
            tesseract_path: Path to tesseract executable (optional)
        """
        if tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = tesseract_path
    
    def extract_text(self, image_path: str, lang: str = "eng") -> str:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to image file
            lang: Language code for OCR (default: "eng")
        
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return None
    
    def extract_text_with_config(self, image_path: str, config: str = "") -> str:
        """
        Extract text with custom Tesseract config
        
        Args:
            image_path: Path to image file
            config: Tesseract configuration string
        
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text with config from {image_path}: {str(e)}")
            return None
    
    def extract_data(self, image_path: str) -> dict:
        """
        Extract structured data from image using Tesseract
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with OCR data (boxes, confidence, text)
        """
        try:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(image)
            return self._parse_tesseract_data(data)
        except Exception as e:
            print(f"Error extracting data from {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def _parse_tesseract_data(data: str) -> dict:
        """Parse Tesseract output data"""
        results = {
            "text": [],
            "confidence": [],
            "boxes": []
        }
        
        for line in data.split('\n')[1:]:
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 12:
                text = parts[11]
                confidence = float(parts[10])
                x = int(parts[6])
                y = int(parts[7])
                w = int(parts[8])
                h = int(parts[9])
                
                if text.strip():
                    results["text"].append(text)
                    results["confidence"].append(confidence)
                    results["boxes"].append((x, y, w, h))
        
        return results
    
    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to image file
            output_path: Optional path to save preprocessed image
        
        Returns:
            Preprocessed image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(threshold)
            
            # Optional: Save preprocessed image
            if output_path:
                cv2.imwrite(output_path, denoised)
            
            return denoised
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def crop_region(self, image_path: str, bbox: Tuple[int, int, int, int], 
                   output_path: Optional[str] = None) -> np.ndarray:
        """
        Crop a region from image based on bounding box
        
        Args:
            image_path: Path to image file
            bbox: Tuple of (x, y, width, height)
            output_path: Optional path to save cropped image
        
        Returns:
            Cropped image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                return None
            
            x, y, w, h = bbox
            cropped = image[y:y+h, x:x+w]
            
            if output_path:
                cv2.imwrite(output_path, cropped)
            
            return cropped
        except Exception as e:
            print(f"Error cropping region from {image_path}: {str(e)}")
            return None

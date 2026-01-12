"""
Name OCR Service

Polls the MongoDB database for records missing the `name` field,
loads the image from disk, crops the name area, runs OCR with
pytesseract, and updates the record with the extracted text.

Run once (process current queue) with:
    py name_ocr_service.py --once

Run continuously with:
    py name_ocr_service.py

Adjust `CROP_BOX` as needed for your card images.
"""

import os
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Tuple

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from PIL import Image
import pytesseract
import re

# Configuration
POLL_INTERVAL = float(os.getenv("NAME_OCR_POLL_INTERVAL", "3"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "pokemon_images")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_images"))

# Crop box for name region: (left, upper, right, lower)
# These are defaults and likely need adjustment per card layout.
# Example values target a top area where the name typically appears.
CROP_BOX = tuple(int(x) for x in os.getenv("NAME_CROP_BOX", "40,10,460,80").split(","))

RETRY_DELAY = 5.0


def connect_db(retries: int = 3, delay: float = RETRY_DELAY) -> MongoClient:
    for attempt in range(1, retries + 1):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # trigger a connection
            client.admin.command("ping")
            print("[DB] Connected to MongoDB")
            return client
        except Exception as e:
            print(f"[DB] Connection attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise


def find_unprocessed(collection, limit: int = 5):
    # Find records where name is missing, null, or empty string
    query = {
        "$or": [
            {"name": None},
            {"name": ""},
            {"name": {"$exists": False}}
        ]
    }
    return list(collection.find(query).limit(limit))


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Keep letters, numbers and spaces; uppercase for consistency
    text = re.sub(r"[^A-Za-z0-9 \-]", "", text)
    return text.upper()


def crop_name_region(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)


def ocr_image(image: Image.Image) -> str:
    try:
        # Convert to grayscale to improve OCR in many cases
        gray = image.convert("L")
        # Use single-line mode psm 7 for names
        config = "--psm 7"
        text = pytesseract.image_to_string(gray, config=config)
        return sanitize_text(text)
    except Exception as e:
        print(f"[OCR] Error during OCR: {e}")
        return ""


def process_record(collection, record) -> None:
    record_id = record.get("_id") or record.get("id")
    filename = record.get("filename")
    filepath = record.get("filepath")

    print(f"[Found] id: {record_id} filename: {filename}")

    # Resolve path
    path = Path(filepath) if filepath else (UPLOAD_DIR / filename)
    if not path.exists():
        # try relative inside UPLOAD_DIR
        alt = UPLOAD_DIR / (filename or "")
        if alt.exists():
            path = alt
        else:
            msg = f"File not found: {path}"
            print(f"[Error] {msg}")
            try:
                collection.update_one({"_id": record_id}, {"$set": {"name_error": msg, "updated_at": datetime.utcnow()}})
            except Exception:
                print("[DB] Failed to update missing-file error")
            return

    try:
        print(f"[Loading] {path}")
        with Image.open(path) as img:
            print("[Cropping] name region")
            cropped = crop_name_region(img, CROP_BOX)
            print("[OCR] extracting text")
            name_text = ocr_image(cropped)

        if not name_text:
            print(f"[OCR] No text extracted for {record_id}")
            collection.update_one({"_id": record_id}, {"$set": {"name": "", "name_error": "ocr_empty", "updated_at": datetime.utcnow()}})
        else:
            print(f"[Extracted] {name_text}")
            collection.update_one({"_id": record_id}, {"$set": {"name": name_text, "updated_at": datetime.utcnow()}})

    except Exception as e:
        print(f"[Error] processing record {record_id}: {e}")
        traceback.print_exc()
        try:
            collection.update_one({"_id": record_id}, {"$set": {"name_error": str(e), "updated_at": datetime.utcnow()}})
        except Exception:
            print("[DB] Failed to update error field for record")


def main(once: bool = False):
    client = connect_db()
    db = client[MONGODB_DB]
    collection = db[COLLECTION_NAME]

    print(f"[Service] Name OCR service started. Polling every {POLL_INTERVAL}s. Crop box={CROP_BOX}")

    try:
        while True:
            try:
                records = find_unprocessed(collection, limit=10)
            except PyMongoError as e:
                print(f"[DB] Error querying for unprocessed images: {e}")
                time.sleep(RETRY_DELAY)
                continue

            if not records:
                print(f"[Waiting] No unprocessed images. Checking again in {POLL_INTERVAL}s.")
                if once:
                    break
                time.sleep(POLL_INTERVAL)
                continue

            for rec in records:
                process_record(collection, rec)

            if once:
                break

            # small pause before next cycle to avoid busy-looping
            time.sleep(POLL_INTERVAL)
    finally:
        client.close()
        print("[Service] Shutting down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Name OCR Service for Pokemon Scanner")
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit")
    args = parser.parse_args()
    main(once=args.once)
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

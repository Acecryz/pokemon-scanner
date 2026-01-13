"""
Lore OCR Service

Polls the MongoDB database for records missing the `lore` field,
loads the image from disk, crops the lore area, runs OCR with
pytesseract, and updates the record with the extracted text.

Run once (process current queue) with:
    py lore_ocr_service.py --once

Run continuously with:
    py lore_ocr_service.py

Adjust `CROP_BOX` as needed for your card images.
"""

import os
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Tuple
import cv2
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from PIL import Image
import pytesseract
import re
DEBUG_OCR = True
DEBUG_OCR_DIR = Path("debug_ocr_inputs")
DEBUG_OCR_DIR.mkdir(exist_ok=True)

# Configuration
POLL_INTERVAL = float(os.getenv("LORE_OCR_POLL_INTERVAL", "3"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
COLLECTION_LORE = os.getenv("MONGODB_COLLECTION", "pokemon_images")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_images"))

# Crop box for lore region: (left, upper, right, lower)
# Lore coordinates: (228, 923, 699, 1016)
CROP_BOX = tuple(int(x) for x in os.getenv("LORE_CROP_BOX", "224,922,697,988").split(","))

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

def upscale(image: Image.Image, scale: int = 2) -> Image.Image:
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.BICUBIC)


def find_unprocessed(collection, limit: int = 5):
    # Find records where lore is missing, null, or empty string
    query = {
        "$or": [
            {"lore": None},
            {"lore": ""},
            {"lore": {"$exists": False}}
        ]
    }
    return list(collection.find(query).limit(limit))


def sanitize_text(text: str) -> str:
    """
    Clean OCR text while preserving natural punctuation and paragraph flow.
    """
    if not text:
        return ""

    # Normalize unicode punctuation to ASCII
    text = text.replace("…", "...")

    # Remove non-printable characters (keep punctuation)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    # Normalize whitespace while preserving line breaks
    lines = [line.strip() for line in text.split("\n")]

    return "\n".join(line for line in lines if line)



def crop_lore_region(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Gentle preprocessing for small serif body text.
    Preserves thin strokes like 'i' and punctuation.
    """
    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Mild denoise only
    img = cv2.fastNlMeansDenoising(img, h=10)

    # Increase contrast without destroying detail
    img = cv2.normalize(
        img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )

    return Image.fromarray(img)


def ocr_image(image: Image.Image) -> str:
    try:
        # Upscale and Preprocess
        image = upscale(image, scale=2)
        image = preprocess_for_ocr(image)

        # OCR Execution
        config = "--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(image, config=config)

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
                collection.update_one({"_id": record_id}, {"$set": {"lore_error": msg, "updated_at": datetime.utcnow()}})
            except Exception:
                print("[DB] Failed to update missing-file error")
            return

    try:
        print(f"[Loading] {path}")
        with Image.open(path) as img:
            print("[Cropping] lore region")
            cropped = crop_lore_region(img, CROP_BOX)
            print("[OCR] extracting text")
            lore_text = ocr_image(cropped)

        if not lore_text:
            print(f"[OCR] No text extracted for {record_id}")
            collection.update_one({"_id": record_id}, {"$set": {"lore": "", "lore_error": "ocr_empty", "updated_at": datetime.utcnow()}})
        else:
            print(f"[Extracted] {lore_text[:100]}")  # Truncate for logging
            collection.update_one({"_id": record_id}, {"$set": {"lore": lore_text, "updated_at": datetime.utcnow()}})

    except Exception as e:
        print(f"[Error] processing record {record_id}: {e}")
        traceback.print_exc()
        try:
            collection.update_one({"_id": record_id}, {"$set": {"lore_error": str(e), "updated_at": datetime.utcnow()}})
        except Exception:
            print("[DB] Failed to update error field for record")


def main(once: bool = False):
    client = connect_db()
    db = client[MONGODB_DB]
    collection = db[COLLECTION_LORE]

    print(f"[Service] Lore OCR service started. Polling every {POLL_INTERVAL}s. Crop box={CROP_BOX}")

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
    parser = argparse.ArgumentParser(description="Lore OCR Service for Pokemon Scanner")
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit")
    args = parser.parse_args()
    main(once=args.once)

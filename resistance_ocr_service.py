"""
Resistance OCR Service

Polls MongoDB for card images missing resistance info (or missing the saved
resistance crop), extracts the resistance region, runs OCR with pytesseract,
saves the crop to disk, and updates the record. Mirrored from
weakness_ocr_service with different crop coordinates.
"""

import argparse
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from PIL import Image
import pytesseract
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLL_INTERVAL = float(os.getenv("RESISTANCE_OCR_POLL_INTERVAL", "3"))
RETRY_DELAY = float(os.getenv("RESISTANCE_OCR_RETRY_DELAY", "5"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
COLLECTION_RESISTANCE = os.getenv("MONGODB_COLLECTION", "pokemon_images")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_images"))
RESISTANCE_OUTPUT_DIR = Path(os.getenv("RESISTANCE_OUTPUT_DIR", "resistances"))
RESISTANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop box for resistance region: (left, upper, right, lower)
CROP_BOX = tuple(int(x) for x in os.getenv("RESISTANCE_CROP_BOX", "214,884,396,911").split(","))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def connect_db(retries: int = 3) -> MongoClient:
    """Connect to MongoDB with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            print("[DB] Connected to MongoDB")
            return client
        except Exception as exc:
            print(f"[DB] Connection attempt {attempt} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(RETRY_DELAY)


def find_unprocessed(collection, limit: int = 10):
    """Fetch records missing resistance text or crop filepath."""
    query = {
        "$or": [
            {"resistance": None},
            {"resistance": ""},
            {"resistance": {"$exists": False}},
            {"resistance_filepath": None},
            {"resistance_filepath": ""},
            {"resistance_filepath": {"$exists": False}},
        ]
    }
    return list(collection.find(query).limit(limit))


def sanitize_text(text: str) -> str:
    """Uppercase + strip non-alphanumeric characters."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9 \-]", "", text)
    return text.upper()


def crop_resistance_region(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop the configured resistance region."""
    return img.crop(box)


def save_cropped_image(img: Image.Image, record_id) -> Path:
    """Persist cropped resistance region for debugging/training."""
    try:
        filename = f"{record_id}.png"
        output_path = RESISTANCE_OUTPUT_DIR / filename
        img.save(output_path, format="PNG")
        return output_path
    except Exception as exc:
        print(f"[Save] Failed to store resistance crop for {record_id}: {exc}")
        return None


def ocr_image(image: Image.Image) -> str:
    """Run pytesseract on the cropped resistance region."""
    try:
        gray = image.convert("L")
        text = pytesseract.image_to_string(gray, config="--psm 7")
        return sanitize_text(text)
    except Exception as exc:
        print(f"[OCR] Error during OCR: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------
def process_record(collection, record) -> None:
    record_id = record.get("id") or record.get("_id")
    filename = record.get("filename")
    filepath = record.get("filepath")

    print(f"[Found] id={record_id} file={filename}")

    path = Path(filepath) if filepath else (UPLOAD_DIR / filename)
    if not path.exists():
        alt = UPLOAD_DIR / (filename or "")
        if alt.exists():
            path = alt
        else:
            msg = f"File not found: {path}"
            print(f"[Error] {msg}")
            collection.update_one(
                {"id": record_id},
                {"$set": {"resistance_error": msg, "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}}
            )
            return

    try:
        with Image.open(path) as img:
            cropped = crop_resistance_region(img, CROP_BOX)
            resistance_crop_path = save_cropped_image(cropped, record_id)

            # Reuse existing resistance text if already present; otherwise OCR
            existing_resistance = record.get("resistance") or ""
            resistance_text = existing_resistance or ocr_image(cropped)

        if not resistance_text:
            print(f"[OCR] empty text for {record_id}")
            update_fields = {
             #   "resistance": "",
              #  "resistance_error": "ocr_empty",
              #  "updated_at": datetime.utcnow()
            }
            if resistance_crop_path:
                update_fields["resistance_filepath"] = str(resistance_crop_path)
            collection.update_one({"id": record_id}, {"$set": update_fields})
            return

        print(f"[Extracted] resistance={resistance_text}")
        update_fields = {
         #   "resistance": resistance_text,
          #  "resistance_error": "",
           # "updated_at": datetime.utcnow()
        }
        if resistance_crop_path:
            update_fields["resistance_filepath"] = str(resistance_crop_path)
        collection.update_one({"id": record_id}, {"$set": update_fields})

    except Exception as exc:
        print(f"[Error] processing {record_id}: {exc}")
        traceback.print_exc()
        collection.update_one(
            {"_id": record_id},
            {"$set": {
                "resistance_error": str(exc),
                "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            }}
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(once: bool = False):
    client = connect_db()
    db = client[MONGODB_DB]
    collection = db[COLLECTION_RESISTANCE]

    print(f"[Service] Resistance OCR started. poll={POLL_INTERVAL}s crop={CROP_BOX}")

    try:
        while True:
            try:
                records = find_unprocessed(collection, limit=10)
            except PyMongoError as exc:
                print(f"[DB] query error: {exc}")
                time.sleep(RETRY_DELAY)
                continue

            if not records:
                print(f"[Waiting] none pending; sleeping {POLL_INTERVAL}s")
                time.sleep(POLL_INTERVAL)
                if once:
                    break
                continue

            for record in records:
                process_record(collection, record)

            if once:
                break

            time.sleep(POLL_INTERVAL)
    finally:
        client.close()
        print("[Service] shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon Resistance OCR Service")
    parser.add_argument("--once", action="store_true", help="Process current queue then exit")
    args = parser.parse_args()
    main(once=args.once)
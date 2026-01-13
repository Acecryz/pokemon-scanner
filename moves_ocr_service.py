"""
Moves OCR Service

Polls MongoDB for card images missing moves info, extracts the cropped
moves region, performs OCR with pytesseract, and updates the record.
Designed to run continuously (or once) as part of the Pokemon Scanner pipeline.
"""

import argparse
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple

from PIL import Image
import pytesseract
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLL_INTERVAL = float(os.getenv("MOVES_OCR_POLL_INTERVAL", "3"))
RETRY_DELAY = float(os.getenv("MOVES_OCR_RETRY_DELAY", "5"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
COLLECTION_MOVES = os.getenv("MONGODB_COLLECTION", "pokemon_images")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_images"))
MOVES_OUTPUT_DIR = Path(os.getenv("MOVES_OUTPUT_DIR", "moves"))
MOVES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop box for moves region: (left, upper, right, lower)
CROP_BOX = tuple(int(x) for x in os.getenv("MOVES_CROP_BOX", "46,519,701,868").split(","))


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
    """Fetch records missing the saved moves crop path."""
    query = {
        "$or": [
            {"moves_filepath": None},
            {"moves_filepath": ""},
            {"moves_filepath": {"$exists": False}},
        ]
    }
    return list(collection.find(query).limit(limit))


def crop_moves_region(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop the configured moves region."""
    return img.crop(box)


def save_cropped_image(img: Image.Image, record_id) -> Path:
    """Persist cropped moves region for debugging/training."""
    try:
        filename = f"{record_id}.png"
        output_path = MOVES_OUTPUT_DIR / filename
        img.save(output_path, format="PNG")
        return output_path
    except Exception as exc:
        print(f"[Save] Failed to store moves crop for {record_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------
def process_record(collection, record) -> None:
    record_id = record.get("_id") or record.get("id")
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
                {"_id": record_id},
                {"$set": {"moves_error": msg, "updated_at": datetime.utcnow()}}
            )
            return

    try:
        with Image.open(path) as img:
            cropped = crop_moves_region(img, CROP_BOX)
            moves_crop_path = save_cropped_image(cropped, record_id)
        update_fields = {
            "updated_at": datetime.utcnow()
        }
        if moves_crop_path:
            update_fields["moves_filepath"] = str(moves_crop_path)
        collection.update_one({"_id": record_id}, {"$set": update_fields})

    except Exception as exc:
        print(f"[Error] processing {record_id}: {exc}")
        traceback.print_exc()
        collection.update_one(
            {"_id": record_id},
            {"$set": {
                "updated_at": datetime.utcnow()
            }}
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(once: bool = False):
    client = connect_db()
    db = client[MONGODB_DB]
    collection = db[COLLECTION_MOVES]

    print(f"[Service] Moves OCR started. poll={POLL_INTERVAL}s crop={CROP_BOX}")

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
                if once:
                    break
                time.sleep(POLL_INTERVAL)
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
    parser = argparse.ArgumentParser(description="Pokemon Moves OCR Service")
    parser.add_argument("--once", action="store_true", help="Process current queue then exit")
    args = parser.parse_args()
    main(once=args.once)

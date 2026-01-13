"""
Weakness OCR Service

Polls MongoDB for card images missing weakness info, extracts the cropped
weakness region, performs OCR with pytesseract, infers the weakness type/color,
and updates the record. Designed to run continuously (or once) as part of the
Pokemon Scanner pipeline.
"""

import argparse
import math
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
import pytesseract
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLL_INTERVAL = float(os.getenv("WEAKNESS_OCR_POLL_INTERVAL", "3"))
RETRY_DELAY = float(os.getenv("WEAKNESS_OCR_RETRY_DELAY", "5"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
COLLECTION_WEAKNESS = os.getenv("MONGODB_COLLECTION", "pokemon_images")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploaded_images"))
WEAKNESS_OUTPUT_DIR = Path(os.getenv("WEAKNESS_OUTPUT_DIR", "weaknesses"))
WEAKNESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop box for weakness region: (left, upper, right, lower)
CROP_BOX = tuple(int(x) for x in os.getenv("WEAKNESS_CROP_BOX", "33,882,206,911").split(","))

# Reference colors (sRGB) for each Pokemon weakness symbol
WEAKNESS_RGB_MAP: Dict[str, Tuple[int, int, int]] = {
    "normal": (168, 167, 122),
    "fire": (238, 129, 48),
    "water": (99, 144, 240),
    "grass": (122, 199, 76),
    "electric": (247, 208, 44),
    "ice": (150, 217, 214),
    "fighting": (194, 46, 40),
    "poison": (163, 62, 161),
    "ground": (226, 191, 101),
    "flying": (168, 145, 236),
    "psychic": (249, 85, 135),
    "bug": (166, 185, 26),
    "rock": (182, 161, 54),
    "ghost": (115, 87, 151),
    "dragon": (111, 53, 252),
    "darkness": (112, 87, 70),
    "metal": (183, 183, 206),
    "fairy": (214, 133, 173),
}

# Text synonyms so OCR mistakes can still map to a canonical weakness value
TEXT_WEAKNESS_MAP = {
    "fire": ["fire", "flame", "burn"],
    "water": ["water", "aqua"],
    "grass": ["grass", "plant", "leaf"],
    "electric": ["electric", "electricity", "lightning", "bolt"],
    "fighting": ["fighting", "fight", "fist"],
    "rock": ["rock", "stone"],
    "ground": ["ground", "earth"],
    "psychic": ["psychic", "mind"],
    "darkness": ["dark", "darkness"],
    "metal": ["metal", "steel"],
    "fairy": ["fairy"],
    "ghost": ["ghost"],
    "dragon": ["dragon"],
    "bug": ["bug"],
    "flying": ["flying", "wind"],
    "ice": ["ice"],
    "poison": ["poison"],
    "normal": ["normal"],
}


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
    """Fetch records that still need weakness data."""
    query = {
        "$or": [
            {"weakness": None},
            {"weakness": ""},
            {"weakness": {"$exists": False}}
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


def crop_weakness_region(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Crop the configured weakness region."""
    return img.crop(box)


def save_cropped_image(img: Image.Image, record_id) -> Path:
    """Persist cropped weakness region for debugging/training."""
    try:
        filename = f"{record_id}.png"
        output_path = WEAKNESS_OUTPUT_DIR / filename
        img.save(output_path, format="PNG")
        return output_path
    except Exception as exc:
        print(f"[Save] Failed to store weakness crop for {record_id}: {exc}")
        return None


def _saturation(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val == 0:
        return 0.0
    return (max_val - min_val) / max_val


def _brightness(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def get_symbol_color(img: Image.Image) -> Tuple[int, int, int]:
    """
    Attempt to isolate the vibrant portion of the weakness symbol so that
    text/background colors do not skew the average.
    """
    img = img.convert("RGB")
    pixels = list(img.getdata())

    vibrant = [
        p for p in pixels
        if 0.2 < _brightness(p) < 0.9 and _saturation(p) > 0.15
    ]
    if not vibrant:
        vibrant = pixels

    vibrant.sort(key=_saturation, reverse=True)
    top_count = max(1, len(vibrant) // 5)
    sample = vibrant[:top_count]

    r = sum(p[0] for p in sample) // len(sample)
    g = sum(p[1] for p in sample) // len(sample)
    b = sum(p[2] for p in sample) // len(sample)
    return (r, g, b)


def closest_color(rgb: Tuple[int, int, int]) -> str:
    """Find the canonical weakness color nearest to rgb."""
    min_dist = float("inf")
    closest = "unknown"
    for key, value in WEAKNESS_RGB_MAP.items():
        dist = math.sqrt(sum((rgb[i] - value[i]) ** 2 for i in range(3)))
        if dist < min_dist:
            min_dist = dist
            closest = key
    return closest


def map_text_to_weakness(text: str) -> str:
    """Map OCR text (with synonyms) to canonical weakness type."""
    if not text:
        return ""
    key = text.strip().lower()
    for weakness, synonyms in TEXT_WEAKNESS_MAP.items():
        if key in synonyms:
            return weakness
    # direct fallback (if OCR already matches the canonical key)
    if key in WEAKNESS_RGB_MAP:
        return key
    return ""


def get_weakness_color(img: Image.Image, weakness_text: str = None) -> str:
    """Combine OCR text and color analysis to determine weakness."""
    color_guess = closest_color(get_symbol_color(img))
    text_guess = map_text_to_weakness(weakness_text or "")
    return text_guess or color_guess


def ocr_image(image: Image.Image) -> str:
    """Run pytesseract on the cropped weakness region."""
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
                {"$set": {"weakness_error": msg, "updated_at": datetime.utcnow()}}
            )
            return

    try:
        with Image.open(path) as img:
            cropped = crop_weakness_region(img, CROP_BOX)
            save_cropped_image(cropped, record_id)
            weakness_text = ocr_image(cropped)

        if not weakness_text:
            print(f"[OCR] empty text for {record_id}")
            collection.update_one(
                {"_id": record_id},
                {"$set": {
                    "weakness": "",
                    "weakness_color": "",
                    "weakness_error": "ocr_empty",
                    "updated_at": datetime.utcnow()
                }}
            )
            return

        weakness_color = get_weakness_color(cropped, weakness_text)

        print(f"[Extracted] weakness={weakness_text} color={weakness_color}")
        collection.update_one(
            {"_id": record_id},
            {"$set": {
                "weakness": weakness_text,
                "weakness_color": weakness_color,
                "updated_at": datetime.utcnow()
            }}
        )

    except Exception as exc:
        print(f"[Error] processing {record_id}: {exc}")
        traceback.print_exc()
        collection.update_one(
            {"_id": record_id},
            {"$set": {
                "weakness_error": str(exc),
                "updated_at": datetime.utcnow()
            }}
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(once: bool = False):
    client = connect_db()
    db = client[MONGODB_DB]
    collection = db[COLLECTION_WEAKNESS]

    print(f"[Service] Weakness OCR started. poll={POLL_INTERVAL}s crop={CROP_BOX}")

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
    parser = argparse.ArgumentParser(description="Pokemon Weakness OCR Service")
    parser.add_argument("--once", action="store_true", help="Process current queue then exit")
    args = parser.parse_args()
    main(once=args.once)

"""
Pokemon Scanner API
HTTP API for image upload and Pokemon data retrieval with MongoDB backend
"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import uvicorn

# Configuration
UPLOAD_DIR = Path("uploaded_images")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pokemon_scanner")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Pokemon Scanner API", version="1.0.0")

# MongoDB client and database
mongo_client = None
db = None


def init_db():
    """Initialize MongoDB connection and create indexes"""
    global mongo_client, db
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Verify connection
        mongo_client.admin.command('ping')
        db = mongo_client[MONGODB_DB]
        
        # Create collection and indexes
        if "pokemon_images" not in db.list_collection_names():
            db.create_collection("pokemon_images")
        
        # Create indexes
        db.pokemon_images.create_index([("id", ASCENDING)], unique=True)
        db.pokemon_images.create_index([("created_at", ASCENDING)])
        
        print("MongoDB connected successfully")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection error: {str(e)}")
        raise


def get_db():
    """Get MongoDB database instance"""
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    return db


@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup"""
    try:
        init_db()
    except Exception as e:
        print(f"Failed to initialize database: {str(e)}")
        print("WARNING: Starting API without MongoDB. Database operations will fail.")
        print("To fix this, either:")
        print("1. Install MongoDB: https://www.mongodb.com/try/download/community")
        print("2. Use MongoDB Atlas: https://www.mongodb.com/cloud/atlas")
        print("3. Set MONGODB_URI env variable with your connection string")


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown"""
    global mongo_client
    if mongo_client:
        mongo_client.close()


@app.post("/")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a Pokemon image and store metadata
    
    Args:
        file: Image file to upload
    
    Returns:
        JSON response with record ID and metadata
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Generate unique ID
        record_id = str(uuid.uuid4())
        
        # Save file to disk
        try:
            file_content = await file.read()
            if len(file_content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty"
                )
            
            filepath = UPLOAD_DIR / f"{record_id}_{file.filename}"
            with open(filepath, "wb") as f:
                f.write(file_content)
        except IOError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}"
            )
        
        # Store metadata in MongoDB
        try:
            database = get_db()
            relative_filepath = str(filepath).replace("\\", "/")
            
            document = {
                "id": record_id,
                "_id": record_id,
                "filename": file.filename,
                "filepath": relative_filepath,
                "uploaded": True,
                "name": None,
                "lore": None,
                "weakness": None,
                "resistance": None,
                "moves": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = database.pokemon_images.insert_one(document)
            
        except Exception as e:
            # Clean up uploaded file if database insert fails
            try:
                filepath.unlink()
            except:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "id": record_id,
                "filename": file.filename,
                "filepath": relative_filepath,
                "message": "File uploaded and document created in MongoDB."
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during upload: {str(e)}"
        )


@app.get("/{record_id}")
async def retrieve_image_data(record_id: str) -> JSONResponse:
    """
    Retrieve Pokemon data and metadata for a specific record
    
    Args:
        record_id: Unique identifier of the uploaded image
    
    Returns:
        JSON response with complete metadata including extracted data
    """
    try:
        database = get_db()
        document = database.pokemon_images.find_one({"_id": record_id})
        
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Record with ID '{record_id}' not found"
            )
        
        # Convert datetime objects to ISO format strings
        if isinstance(document.get("created_at"), datetime):
            document["created_at"] = document["created_at"].isoformat()
        if isinstance(document.get("updated_at"), datetime):
            document["updated_at"] = document["updated_at"].isoformat()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=document
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@app.patch("/{record_id}")
async def update_record(record_id: str, updates: Dict[str, Any]) -> JSONResponse:
    """
    Update metadata for a specific record (used by OCR services)
    
    Args:
        record_id: Unique identifier of the record to update
        updates: Dictionary of fields to update
    
    Returns:
        JSON response with updated metadata
    """
    try:
        database = get_db()
        
        # Verify record exists
        if database.pokemon_images.find_one({"_id": record_id}) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Record with ID '{record_id}' not found"
            )
        
        # Allowed fields for update
        allowed_fields = {
            "name", "lore", "weakness_filepath", 
            "resistance_filepath", "moves_filepath"
        }
        
        # Filter updates to allowed fields only
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not filtered_updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )
        
        # Add updated_at timestamp
        filtered_updates["updated_at"] = datetime.utcnow()
        
        # Update document
        database.pokemon_images.update_one(
            {"_id": record_id},
            {"$set": filtered_updates}
        )
        
        # Fetch and return updated record
        document = database.pokemon_images.find_one({"_id": record_id})
        
        # Convert datetime objects to ISO format strings
        if isinstance(document.get("created_at"), datetime):
            document["created_at"] = document["created_at"].isoformat()
        if isinstance(document.get("updated_at"), datetime):
            document["updated_at"] = document["updated_at"].isoformat()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=document
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@app.get("/")
async def health_check() -> JSONResponse:
    """Health check endpoint"""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "Pokemon Scanner API",
            "version": "1.0.0"
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

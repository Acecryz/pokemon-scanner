"""
Pokemon Scanner API
HTTP API for image upload and Pokemon data retrieval with MongoDB backend
"""

import os
import uuid
import threading
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from enum import Enum

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


class OCRServiceName(str, Enum):
    """Enum for OCR service names"""
    name = "name"
    weakness = "weakness"
    resistance = "resistance"
    moves = "moves"
    lore = "lore"


# OCR Service Manager
class OCRServiceManager:
    """Manages OCR service processes"""
    def __init__(self):
        self.services = {
            "name": {"process": None, "running": False},
            "weakness": {"process": None, "running": False},
            "resistance": {"process": None, "running": False},
            "moves": {"process": None, "running": False},
            "lore": {"process": None, "running": False},
        }
    
    def start_service(self, service_name: str) -> bool:
        """Start an OCR service"""
        if service_name not in self.services:
            return False
        
        if self.services[service_name]["running"]:
            return False  # Already running
        
        try:
            script_name = f"{service_name}_ocr_service.py"
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.services[service_name]["process"] = process
            self.services[service_name]["running"] = True
            return True
        except Exception as e:
            print(f"Error starting {service_name} service: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop an OCR service"""
        if service_name not in self.services:
            return False
        
        if not self.services[service_name]["running"]:
            return False  # Not running
        
        try:
            process = self.services[service_name]["process"]
            if process:
                process.terminate()
                process.wait(timeout=5)
            self.services[service_name]["process"] = None
            self.services[service_name]["running"] = False
            return True
        except Exception as e:
            print(f"Error stopping {service_name} service: {e}")
            # Force kill if terminate didn't work
            try:
                if process:
                    process.kill()
            except:
                pass
            self.services[service_name]["process"] = None
            self.services[service_name]["running"] = False
            return True
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all services"""
        status = {}
        for name, info in self.services.items():
            # Check if process is still alive
            if info["running"] and info["process"]:
                if info["process"].poll() is not None:
                    # Process has terminated
                    info["running"] = False
                    info["process"] = None
            status[name] = info["running"]
        return status
    
    def stop_all(self):
        """Stop all running services"""
        for service_name in list(self.services.keys()):
            if self.services[service_name]["running"]:
                self.stop_service(service_name)

# Global service manager instance
service_manager = OCRServiceManager()


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
    """Close MongoDB connection and stop all OCR services on shutdown"""
    global mongo_client
    # Stop all OCR services
    service_manager.stop_all()
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
            
            now_utc = datetime.now(timezone.utc)
            
            document = {
                "_id": record_id,  # Use id as MongoDB _id
                "id": record_id,
                "filename": file.filename,
                "filepath": relative_filepath,
               # "uploaded": True,
                "name": None,
                "lore": None,
                "weakness_filepath": None,
                "resistance_filepath": None,
                "moves_filepath": None,
                "created_at": now_utc.strftime('%Y-%m-%d %H:%M:%S'),
                "updated_at": now_utc.strftime('%Y-%m-%d %H:%M:%S')
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
        document = database.pokemon_images.find_one({"id": record_id})
        
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Record with ID '{record_id}' not found"
            )
        
        # Convert to dict and remove _id from response
        document = dict(document)
        document.pop("_id", None)
        
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

    now_utc = datetime.now(timezone.utc)

    try:
        database = get_db()
        
        # Verify record exists
        if database.pokemon_images.find_one({"id": record_id}) is None:
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
        filtered_updates["updated_at"] = now_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update document
        database.pokemon_images.update_one(
            {"id": record_id},
            {"$set": filtered_updates}
        )
        
        # Fetch and return updated record
        document = database.pokemon_images.find_one({"id": record_id})
        
        # Convert to dict and remove _id from response
        document = dict(document)
        document.pop("_id", None)
        
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


@app.post("/services/{service_name}/start")
async def start_ocr_service(service_name: OCRServiceName) -> JSONResponse:
    """
    Start an OCR service
    
    Args:
        service_name: Name of the service (name, weakness, resistance, moves, lore)
    
    Returns:
        JSON response with service status
    """
    service_name_str = service_name.value
    
    if service_manager.services[service_name_str]["running"]:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "service": service_name_str,
                "status": "already_running",
                "message": f"{service_name_str} OCR service is already running"
            }
        )
    
    success = service_manager.start_service(service_name_str)
    if success:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "service": service_name_str,
                "status": "started",
                "message": f"{service_name_str} OCR service started successfully"
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start {service_name_str} OCR service"
        )


@app.post("/services/{service_name}/stop")
async def stop_ocr_service(service_name: OCRServiceName) -> JSONResponse:
    """
    Stop an OCR service
    
    Args:
        service_name: Name of the service (name, weakness, resistance, moves, lore)
    
    Returns:
        JSON response with service status
    """
    service_name_str = service_name.value
    
    if not service_manager.services[service_name_str]["running"]:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "service": service_name_str,
                "status": "not_running",
                "message": f"{service_name_str} OCR service is not running"
            }
        )
    
    success = service_manager.stop_service(service_name_str)
    if success:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "service": service_name_str,
                "status": "stopped",
                "message": f"{service_name_str} OCR service stopped successfully"
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop {service_name_str} OCR service"
        )


@app.get("/services/status")
async def get_services_status() -> JSONResponse:
    """
    Get status of all OCR services
    
    Returns:
        JSON response with status of all services
    """
    status_dict = service_manager.get_status()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "services": status_dict,
            "message": "Service status retrieved successfully"
        }
    )


@app.get("/services/{service_name}/status")
async def get_service_status(service_name: OCRServiceName) -> JSONResponse:
    """
    Get status of a specific OCR service
    
    Args:
        service_name: Name of the service (name, weakness, resistance, moves, lore)
    
    Returns:
        JSON response with service status
    """
    service_name_str = service_name.value
    is_running = service_manager.services[service_name_str]["running"]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "service": service_name_str,
            "running": is_running,
            "status": "running" if is_running else "stopped"
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

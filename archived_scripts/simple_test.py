# Simple test to check S3 folder access and metadata extraction
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uuid
import logging
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Test API", description="Test S3 access and metadata extraction")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ImageFolderProcessRequest(BaseModel):
    s3_folder_url: str
    
    @field_validator('s3_folder_url')
    @classmethod
    def validate_s3_folder_url(cls, v):
        if not v.startswith('s3://'):
            raise ValueError('s3_folder_url must start with s3://')
        return v

class ImageProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[str] = None
    processed_images: Optional[int] = None
    total_images: Optional[int] = None
    successful_images: Optional[int] = None
    failed_images: Optional[int] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None

# In-memory task storage
tasks = {}

def list_s3_images(s3_folder_url: str) -> List[str]:
    """List all images in S3 folder"""
    try:
        # Parse S3 URL
        s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
        bucket_name = s3_parts[0]
        folder_prefix = s3_parts[1] + '/' if len(s3_parts) > 1 else ''
        
        # List objects in folder
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
        
        if 'Contents' not in response:
            raise ValueError("No files found in the specified folder")
        
        # Filter for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_objects = [
            obj['Key'] for obj in response['Contents'] 
            if any(obj['Key'].lower().endswith(ext) for ext in image_extensions)
        ]
        
        if not image_objects:
            raise ValueError("No image files found in the folder")
        
        logger.info(f"Found {len(image_objects)} images in folder")
        return sorted(image_objects)
        
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise

def extract_metadata_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """Download image from S3 and extract metadata"""
    try:
        s3_client = boto3.client('s3')
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            s3_client.download_fileobj(bucket, key, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Extract metadata
            with Image.open(tmp_path) as img:
                exif_data = img._getexif()
                
            metadata = {
                'filename': os.path.basename(key),
                's3_key': key,
                'timestamp': None,
                'latitude': None,
                'longitude': None,
                'altitude': None,
                'image_size': None
            }
            
            # Get image size
            with Image.open(tmp_path) as img:
                metadata['image_size'] = f"{img.width}x{img.height}"
            
            if exif_data:
                # Extract timestamp
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'DateTime':
                        try:
                            metadata['timestamp'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
                        except:
                            pass
                
                # Extract GPS data
                if 'GPSInfo' in exif_data:
                    gps_data = exif_data['GPSInfo']
                    
                    def convert_to_degrees(value):
                        if isinstance(value, (list, tuple)) and len(value) == 3:
                            d, m, s = value
                            return float(d) + float(m)/60 + float(s)/3600
                        return float(value) if value else 0
                    
                    if 2 in gps_data and 4 in gps_data:  # Latitude and Longitude
                        lat = convert_to_degrees(gps_data[2])
                        lon = convert_to_degrees(gps_data[4])
                        
                        # Check for hemisphere
                        if 1 in gps_data and gps_data[1] == 'S':
                            lat = -lat
                        if 3 in gps_data and gps_data[3] == 'W':
                            lon = -lon
                            
                        metadata['latitude'] = lat
                        metadata['longitude'] = lon
                    
                    if 6 in gps_data:  # Altitude
                        metadata['altitude'] = float(gps_data[6])
            
            return metadata
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error extracting metadata from {key}: {str(e)}")
        return {
            'filename': os.path.basename(key),
            's3_key': key,
            'error': str(e)
        }

def process_folder_simple(task_id: str, s3_folder_url: str):
    """Process folder and extract metadata"""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = "Listing images..."
        
        # Parse S3 URL
        s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
        bucket_name = s3_parts[0]
        
        # List images
        image_keys = list_s3_images(s3_folder_url)
        tasks[task_id]["total_images"] = len(image_keys)
        
        results = []
        successful = 0
        failed = 0
        
        # Process each image
        for i, image_key in enumerate(image_keys):
            try:
                tasks[task_id]["progress"] = f"Processing image {i+1}/{len(image_keys)}"
                
                metadata = extract_metadata_from_s3(bucket_name, image_key)
                results.append(metadata)
                
                if 'error' not in metadata:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_key}: {str(e)}")
                results.append({
                    'filename': os.path.basename(image_key),
                    's3_key': image_key,
                    'error': str(e)
                })
                failed += 1
            
            # Update progress
            tasks[task_id]["processed_images"] = i + 1
            tasks[task_id]["successful_images"] = successful
            tasks[task_id]["failed_images"] = failed
        
        # Final status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = f"Completed! {successful} successful, {failed} failed"
        tasks[task_id]["results"] = results
        
        logger.info(f"Task {task_id} completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.post("/process-folder", response_model=ImageProcessResponse)
async def process_folder_endpoint(request: ImageFolderProcessRequest, background_tasks: BackgroundTasks):
    """Process images from an S3 folder"""
    try:
        task_id = str(uuid.uuid4())
        
        tasks[task_id] = {
            "status": "queued",
            "progress": None,
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_images": None,
            "results": [],
            "error": None,
            "created_at": datetime.now().isoformat()
        }
        
        background_tasks.add_task(process_folder_simple, task_id, request.s3_folder_url)
        
        return ImageProcessResponse(
            task_id=task_id,
            status="queued",
            message="Image processing started. Use /status/{task_id} to check progress."
        )
        
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data["progress"],
        processed_images=task_data["processed_images"],
        total_images=task_data["total_images"],
        successful_images=task_data["successful_images"],
        failed_images=task_data["failed_images"],
        results=task_data["results"],
        error=task_data["error"]
    )

@app.get("/")
async def root():
    return {"message": "Simple Test API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

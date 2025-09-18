# QR Processing API - Simplified Version
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from typing import Optional, Dict, Any, List
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from inference_sdk import InferenceHTTPClient
from supabase import create_client, Client
import logging
import base64
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import numpy as np
from pyzbar import pyzbar
import requests
import tempfile
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "MQ1Wd6PJGMPBMCvxsCS6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "rcs-k9i1w")
QR_WORKFLOW_ID = os.getenv("QR_WORKFLOW_ID", "qr-workflow")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("SUPABASE_URL and SUPABASE_KEY not found - database features will be disabled")
    supabase = None
else:
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")

# Note: AWS credentials are handled automatically via environment or IAM role

# Initialize Roboflow client (try local server first, fallback to cloud)
def get_roboflow_client():
    """Get Roboflow client, preferring local inference server like image_processing_api"""
    try:
        # Try local inference server first (same as image_processing_api)
        import requests
        response = requests.get("http://localhost:9001/", timeout=2)
        logger.info("Using local inference server at localhost:9001")
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=ROBOFLOW_API_KEY
        )
    except Exception:
        logger.info("Local inference server not available, using cloud API")
        return InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )

app = FastAPI(title="QR Processing API", description="Simple API for processing QR codes from S3 images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Simplified Pydantic models
class QRProcessRequest(BaseModel):
    s3_folder_url: str
    rerun: bool = False  # Force reprocessing, ignore cache
    
    @field_validator('s3_folder_url')
    @classmethod
    def validate_s3_url(cls, v):
        if not v or not v.startswith('s3://'):
            raise ValueError('Must be a valid S3 URL starting with s3://')
        return v

class QRResult(BaseModel):
    content: str
    confidence: float

class ImageQRResult(BaseModel):
    image_name: str
    image_url: str  # Signed S3 URL for the image
    image_coordinates: Optional[Dict[str, float]] = None
    qr_results: List[QRResult] = []
    success: bool
    error: Optional[str] = None

class QRProcessResponse(BaseModel):
    success: bool
    message: str
    total_images: int
    processed_images: int
    results: List[ImageQRResult] = []
    error: Optional[str] = None

def list_images_in_s3_folder(s3_folder_url: str) -> List[str]:
    """List all image files in an S3 folder"""
    try:
        # Parse s3://bucket/folder format
        parts = s3_folder_url[5:].split('/', 1)
        bucket = parts[0]
        folder_prefix = parts[1] if len(parts) > 1 else ''
        
        # Ensure folder prefix ends with /
        if folder_prefix and not folder_prefix.endswith('/'):
            folder_prefix += '/'
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # List objects in the folder
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=folder_prefix,
            Delimiter='/'
        )
        
        image_urls = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Filter for image files
                if key.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                    image_urls.append(f"s3://{bucket}/{key}")
        
        logger.info(f"Found {len(image_urls)} images in folder: {s3_folder_url}")
        return image_urls
        
    except Exception as e:
        logger.error(f"Failed to list images in folder {s3_folder_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list images: {str(e)}")

def download_image_from_s3(s3_url: str) -> str:
    """Download image from S3 (exactly like image_processing_api)"""
    try:
        # Parse s3://bucket/key format
        parts = s3_url[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Create S3 client (same as image_processing_api)
        s3_client = boto3.client('s3')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Download the image (exactly like image_processing_api)
        s3_client.download_file(bucket, key, temp_path)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to download image {s3_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image download failed: {str(e)}")

def get_s3_signed_url(s3_url: str) -> str:
    """Convert S3 URL to signed HTTPS URL"""
    try:
        # Parse s3://bucket/key format
        parts = s3_url[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        # Create S3 client dynamically (same approach as image_processing_api)
        s3_client = boto3.client('s3')
        
        # Generate signed URL
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )
        return signed_url
        
    except Exception as e:
        logger.error(f"Failed to generate signed URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 URL processing failed: {str(e)}")

def extract_gps_from_image(s3_url: str) -> Optional[Dict[str, float]]:
    """Extract GPS coordinates from image EXIF data"""
    try:
        # Get signed URL and download image
        https_url = get_s3_signed_url(s3_url)
        response = requests.get(https_url)
        response.raise_for_status()
        
        # Open image and extract EXIF
        image = Image.open(io.BytesIO(response.content))
        exif_dict = image.getexif()
        
        if exif_dict:
            # Extract GPS data using GPS IFD (the correct method)
            gps_ifd = exif_dict.get_ifd(0x8825)  # GPS IFD tag
            if gps_ifd:
                logger.info(f"📍 Found GPS data in image")
                
                # Get GPS coordinates using numeric tags
                gps_latitude = gps_ifd.get(2)      # GPSLatitude
                gps_latitude_ref = gps_ifd.get(1)  # GPSLatitudeRef
                gps_longitude = gps_ifd.get(4)     # GPSLongitude  
                gps_longitude_ref = gps_ifd.get(3) # GPSLongitudeRef
                
                if gps_latitude and gps_longitude:
                    lat = convert_to_degrees(gps_latitude)
                    lon = convert_to_degrees(gps_longitude)
                    
                    # Apply hemisphere corrections
                    if gps_latitude_ref and gps_latitude_ref.upper() == 'S':
                        lat = -lat
                    if gps_longitude_ref and gps_longitude_ref.upper() == 'W':
                        lon = -lon
                    
                    # Validate coordinates are reasonable
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        logger.info(f"📍 GPS coordinates extracted: {lat:.6f}, {lon:.6f}")
                        return {'latitude': lat, 'longitude': lon}
                    else:
                        logger.warning(f"Invalid GPS coordinates: lat={lat}, lon={lon}")
                        return None
                else:
                    logger.info("📍 GPS data found but latitude/longitude missing")
                    return None
            else:
                logger.info("📍 No GPS IFD found in image EXIF")
                return None
        else:
            logger.info("📍 No EXIF data found in image")
            return None
        
    except Exception as e:
        logger.warning(f"Could not extract GPS data: {str(e)}")
        return None

def convert_to_degrees(value):
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        d, m, s = value
        return float(d) + float(m)/60 + float(s)/3600
    return 0.0

def decode_qr_from_base64(base64_string: str) -> Optional[str]:
    """Decode QR code from base64 image string with enhanced processing"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale first (better for QR detection)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Try multiple processing approaches
        processing_methods = [
            # Method 1: Original image
            img_array,
            
            # Method 2: Enhanced contrast
            cv2.equalizeHist(img_array),
            
            # Method 3: Gaussian blur to reduce noise
            cv2.GaussianBlur(img_array, (3, 3), 0),
            
            # Method 4: Morphological operations to clean up
            cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)),
            
            # Method 5: Threshold to binary
            cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            
            # Method 6: Adaptive threshold
            cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        ]
        
        # Try each processing method
        for i, processed_img in enumerate(processing_methods):
            try:
                qr_codes = pyzbar.decode(processed_img)
                if qr_codes:
                    content = qr_codes[0].data.decode('utf-8')
                    logger.info(f"QR decoded successfully using method {i+1}: {content[:50]}...")
                    return content
            except Exception as method_error:
                logger.debug(f"Method {i+1} failed: {str(method_error)}")
                continue
        
        # If all methods fail, try resizing the image
        try:
            # Scale up the image 2x for better detection
            height, width = img_array.shape
            scaled_img = cv2.resize(img_array, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            qr_codes = pyzbar.decode(scaled_img)
            if qr_codes:
                content = qr_codes[0].data.decode('utf-8')
                logger.info(f"QR decoded successfully using scaling: {content[:50]}...")
                return content
        except Exception as scale_error:
            logger.debug(f"Scaling method failed: {str(scale_error)}")
        
        logger.warning("All QR decoding methods failed")
        return None
        
    except Exception as e:
        logger.warning(f"Could not decode QR code: {str(e)}")
        return None

@app.get("/")
async def root():
    return {
        "message": "QR Processing API", 
        "version": "1.0.0",
        "endpoints": {
            "/process-qr": "POST - Process QR codes from S3 image URL",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def generate_folder_name_from_timestamp_standalone(timestamp_str: str) -> str:
    """Generate folder name from timestamp in format 'Run [Month] [Date] [time in 12 hour]'"""
    try:
        # Parse EXIF timestamp format: "2025:09:15 11:38:20"
        dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
        
        # Format: Run [Month] [Date] [time in 12 hour]
        # Example: "Run September 15 11:38 AM"
        month_name = dt.strftime('%B')  # Full month name
        day = dt.strftime('%d').lstrip('0')  # Day without leading zero
        time_12hr = dt.strftime('%I:%M %p').lstrip('0')  # 12-hour time without leading zero
        
        folder_name = f"Run {month_name} {day} {time_12hr}"
        return folder_name
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing timestamp for folder name: {e}")
        # Fallback to original naming if timestamp parsing fails
        return f"Run {datetime.now().strftime('%B %d %I:%M %p')}"

def extract_image_metadata_standalone(image_path: str) -> Dict[str, Any]:
    """Extract timestamp and GPS coordinates from image EXIF data (standalone version)"""
    try:
        with Image.open(image_path) as img:
            exif_dict = img.getexif()
            
        metadata = {}
        
        # Extract timestamp
        timestamp = exif_dict.get(306)  # DateTime tag
        if timestamp:
            metadata['timestamp'] = timestamp
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Error extracting metadata from {image_path}: {str(e)}")
        return {}

def generate_folder_name_from_s3_folder(s3_folder_url: str) -> str:
    """Generate folder name from first image in S3 folder for unprocessed folders"""
    try:
        # Parse S3 URL to get bucket and prefix
        s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
        bucket = s3_parts[0]
        folder_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
        
        # List objects in the folder to find the first image
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=folder_prefix,
            MaxKeys=10  # Get first few objects to find an image
        )
        
        if 'Contents' not in response:
            return s3_folder_url.rstrip('/').split('/')[-1]  # Fallback to original name
        
        # Find first image file
        first_image_key = None
        for obj in response['Contents']:
            key = obj['Key']
            if key.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                first_image_key = key
                break
        
        if not first_image_key:
            return s3_folder_url.rstrip('/').split('/')[-1]  # Fallback to original name
        
        # Download first image temporarily to extract metadata
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            s3_client.download_file(bucket, first_image_key, temp_file.name)
            
            # Extract metadata from the image
            metadata = extract_image_metadata_standalone(temp_file.name)
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if metadata.get('timestamp'):
                return generate_folder_name_from_timestamp_standalone(metadata['timestamp'])
            else:
                return s3_folder_url.rstrip('/').split('/')[-1]  # Fallback to original name
                
    except Exception as e:
        logger.warning(f"Error generating folder name from S3 folder {s3_folder_url}: {e}")
        return s3_folder_url.rstrip('/').split('/')[-1]  # Fallback to original name

@app.get("/s3-folders")
async def list_s3_folders(bucket: str = "rcsstoragebucket", prefix: str = "fh_sync/bf03aad1-5c1a-464d-8bda-2eac7aeec67f/e1ab4550-4b97-4273-bee4-bccfe1eb87d9/media/"):
    """
    List available folders in S3 for QR code processing
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # List objects in the bucket with the given prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"S3 access error: {str(e)}")
        
        # Extract folder names from CommonPrefixes
        s3_folders = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                folder_path = prefix_info['Prefix']
                # Extract just the folder name (e.g., "0152dbc0-e46a-4bb7-adb7-da2a779c53df" from the path)
                folder_name = folder_path.rstrip('/').split('/')[-1]
                s3_url = f"s3://{bucket}/{folder_path}"
                
                # Generate proper folder name from first image timestamp
                display_folder_name = generate_folder_name_from_s3_folder(s3_url)
                
                # Count images in this folder
                try:
                    image_urls = list_images_in_s3_folder(s3_url)
                    total_images = len(image_urls)
                except Exception as e:
                    logger.warning(f"Could not count images in {s3_url}: {str(e)}")
                    total_images = 0
                
                # Check if folder is cached
                is_processed = False
                processing_status = "unprocessed"
                task_id = None
                processed_images = 0
                total_qr_codes = 0
                
                if supabase:
                    cached_folder = check_folder_in_database(s3_url)
                    if cached_folder:
                        is_processed = cached_folder.get('status') == 'completed'
                        processing_status = cached_folder.get('status', 'unknown')
                        task_id = cached_folder.get('task_id')
                        processed_images = cached_folder.get('processed_images', 0)
                        total_qr_codes = cached_folder.get('total_qr_codes', 0)
                
                s3_folders.append({
                    "folder_name": display_folder_name,
                    "s3_url": s3_url,
                    "s3_path": folder_path,
                    "total_images": total_images,
                    "has_images": total_images > 0,
                    "is_processed": is_processed,
                    "processing_status": processing_status,
                    "task_id": task_id,
                    "processed_images": processed_images,
                    "total_qr_codes": total_qr_codes
                })
        
        # Sort by timestamp (same as image_processing_api)
        def get_folder_timestamp(folder_info):
            try:
                folder_name = folder_info['folder_name']
                if folder_name.startswith('Run '):
                    # Parse "Run September 10 6:40 AM" format
                    try:
                        # Remove "Run " prefix
                        name_part = folder_name[4:]
                        # Split into parts: "September 10 6:40 AM"
                        parts = name_part.split()
                        if len(parts) >= 4:
                            month_name = parts[0]
                            day = parts[1]
                            time_part = ' '.join(parts[2:])  # "6:40 AM"
                            
                            # Convert to datetime for sorting
                            current_year = datetime.now().year
                            dt_str = f"{current_year} {month_name} {day} {time_part}"
                            dt = datetime.strptime(dt_str, "%Y %B %d %I:%M %p")
                            return dt
                    except Exception as e:
                        logger.warning(f"Error parsing folder name timestamp '{folder_name}': {e}")
                
                # Fallback: return epoch time (will sort to beginning)
                return datetime(1970, 1, 1)
                
            except Exception as e:
                logger.warning(f"Error extracting timestamp for folder: {e}")
                return datetime(1970, 1, 1)
        
        # Sort by timestamp (newest first, same as image_processing_api)
        s3_folders.sort(key=get_folder_timestamp, reverse=True)
        
        return {
            "bucket": bucket,
            "prefix": prefix,
            "folders": s3_folders,
            "total_folders": len(s3_folders),
            "folders_with_images": len([f for f in s3_folders if f['has_images']])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing S3 folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing S3 folders: {str(e)}")

# Database helper functions
def check_folder_in_database(s3_folder_url: str) -> Optional[Dict[str, Any]]:
    """Check if folder has been processed before"""
    if not supabase:
        return None
    
    try:
        result = supabase.table('qr_processed_folders').select('*').eq('s3_input_folder_url', s3_folder_url).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.warning(f"Error checking folder in database: {str(e)}")
        return None

def get_cached_folder_results(task_id: str) -> List[Dict[str, Any]]:
    """Get cached results for a processed folder"""
    if not supabase:
        return []
    
    try:
        result = supabase.table('qr_processed_images').select('*').eq('task_id', task_id).order('filename').execute()
        return result.data
    except Exception as e:
        logger.warning(f"Error getting cached results: {str(e)}")
        return []

def create_folder_record(task_id: str, folder_name: str, s3_folder_url: str, total_images: int) -> bool:
    """Create a new folder processing record"""
    if not supabase:
        return False
    
    try:
        supabase.table('qr_processed_folders').insert({
            'task_id': task_id,
            'folder_name': folder_name,
            's3_input_folder_url': s3_folder_url,
            'status': 'processing',
            'total_images': total_images,
            'processing_started_at': datetime.now().isoformat()
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error creating folder record: {str(e)}")
        return False

def update_folder_record(task_id: str, **updates) -> bool:
    """Update folder processing record"""
    if not supabase:
        return False
    
    try:
        supabase.table('qr_processed_folders').update(updates).eq('task_id', task_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error updating folder record: {str(e)}")
        return False

def store_image_result(task_id: str, s3_image_url: str, image_result: ImageQRResult) -> bool:
    """Store individual image processing result in database"""
    if not supabase:
        return False
    
    try:
        # Convert QR results to JSON-serializable format
        qr_results_json = [
            {
                'content': qr.content,
                'confidence': qr.confidence
            }
            for qr in image_result.qr_results
        ]
        
        supabase.table('qr_processed_images').insert({
            'task_id': task_id,
            'filename': image_result.image_name,
            's3_input_url': s3_image_url,
            'image_signed_url': image_result.image_url,
            'timestamp': datetime.now().isoformat(),
            'latitude': image_result.image_coordinates.get('latitude') if image_result.image_coordinates else None,
            'longitude': image_result.image_coordinates.get('longitude') if image_result.image_coordinates else None,
            'qr_results': qr_results_json,
            'processing_status': 'success' if image_result.success else 'failed',
            'error_message': image_result.error,
            'processed_at': datetime.now().isoformat()
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error storing image result: {str(e)}")
        return False

def convert_cached_result_to_image_qr_result(cached_result: Dict[str, Any]) -> ImageQRResult:
    """Convert database record back to ImageQRResult with fresh signed URL"""
    qr_results = []
    if cached_result.get('qr_results'):
        for qr_data in cached_result['qr_results']:
            qr_results.append(QRResult(
                content=qr_data['content'],
                confidence=qr_data['confidence']
            ))
    
    image_coordinates = None
    if cached_result.get('latitude') and cached_result.get('longitude'):
        image_coordinates = {
            'latitude': float(cached_result['latitude']),
            'longitude': float(cached_result['longitude'])
        }
    
    # Generate fresh signed URL since cached ones expire after 1 hour
    fresh_signed_url = ""
    s3_input_url = cached_result.get('s3_input_url', '')
    if s3_input_url:
        try:
            fresh_signed_url = get_s3_signed_url(s3_input_url)
            logger.info(f"Generated fresh signed URL for cached result: {cached_result['filename']}")
        except Exception as e:
            logger.warning(f"Failed to generate fresh signed URL for {cached_result['filename']}: {str(e)}")
            fresh_signed_url = s3_input_url  # Fallback to original S3 URL
    
    return ImageQRResult(
        image_name=cached_result['filename'],
        image_url=fresh_signed_url,
        image_coordinates=image_coordinates,
        qr_results=qr_results,
        success=cached_result.get('processing_status') == 'success',
        error=cached_result.get('error_message')
    )

async def process_single_image(s3_image_url: str) -> ImageQRResult:
    """Process QR codes from a single S3 image"""
    try:
        image_name = s3_image_url.split('/')[-1]
        logger.info(f"Processing image: {image_name}")
        
        # Download image from S3 (exactly like image_processing_api)
        local_image_path = download_image_from_s3(s3_image_url)
        
        try:
            # Extract GPS coordinates from image EXIF data
            image_coordinates = extract_gps_from_image(s3_image_url)
            
            # Get Roboflow client (local server preferred, like image_processing_api)
            roboflow_client = get_roboflow_client()
            
            # Call Roboflow workflow with local file
            result = roboflow_client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=QR_WORKFLOW_ID,
                images={"image": local_image_path}
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(local_image_path)
            except:
                pass
        
        # Process results
        qr_results = []
        if isinstance(result, list) and len(result) > 0:
            workflow_result = result[0]
            
            crop_outputs = workflow_result.get('crop_output', [])
            predictions = workflow_result.get('model_predictions', {}).get('predictions', [])
            
            logger.info(f"Found {len(crop_outputs)} QR code detections in {image_name}")
            
            for i, crop_base64 in enumerate(crop_outputs):
                qr_content = decode_qr_from_base64(crop_base64)
                if qr_content:
                    confidence = predictions[i].get('confidence', 0.0) if i < len(predictions) else 0.0
                    qr_results.append(QRResult(content=qr_content, confidence=confidence))
                    logger.info(f"✅ Decoded QR in {image_name}: {qr_content[:50]}...")
        
        # Generate signed URL for the image
        image_signed_url = get_s3_signed_url(s3_image_url)
        
        return ImageQRResult(
            image_name=image_name,
            image_url=image_signed_url,
            image_coordinates=image_coordinates,
            qr_results=qr_results,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing image {s3_image_url}: {str(e)}")
        # Try to generate signed URL even for failed images
        try:
            image_signed_url = get_s3_signed_url(s3_image_url)
        except:
            image_signed_url = s3_image_url  # Fallback to original URL
            
        return ImageQRResult(
            image_name=s3_image_url.split('/')[-1],
            image_url=image_signed_url,
            image_coordinates=None,
            qr_results=[],
            success=False,
            error=str(e)
        )

@app.post("/process-qr", response_model=QRProcessResponse)
async def process_qr(request: QRProcessRequest):
    """Process QR codes from all images in S3 folder using Roboflow workflow"""
    try:
        logger.info(f"Processing QR for folder: {request.s3_folder_url}")
        
        # Check if folder has been processed before (unless rerun is requested)
        if not request.rerun:
            cached_folder = check_folder_in_database(request.s3_folder_url)
            if cached_folder and cached_folder.get('status') == 'completed':
                logger.info(f"Found cached results for folder: {request.s3_folder_url}")
                
                # Get cached results
                cached_results = get_cached_folder_results(cached_folder['task_id'])
                if cached_results:
                    # Convert cached results to API format
                    results = [convert_cached_result_to_image_qr_result(result) for result in cached_results]
                    total_qr_codes = sum(len(result.qr_results) for result in results)
                    
                    logger.info(f"Returning {len(results)} cached results with {total_qr_codes} QR codes")
                    return QRProcessResponse(
                        success=True,
                        message=f"Retrieved cached results: {len(results)} images, {total_qr_codes} QR codes",
                        total_images=len(results),
                        processed_images=len([r for r in results if r.success]),
                        results=results
                    )
        elif request.rerun:
            # If rerun is requested, delete existing cache
            logger.info(f"Rerun requested - clearing cache for folder: {request.s3_folder_url}")
            cached_folder = check_folder_in_database(request.s3_folder_url)
            if cached_folder and supabase:
                # Delete existing cache
                supabase.table('qr_processed_images').delete().eq('task_id', cached_folder['task_id']).execute()
                supabase.table('qr_processed_folders').delete().eq('task_id', cached_folder['task_id']).execute()
                logger.info(f"Deleted existing cache for folder: {request.s3_folder_url}")
        
        # No cache found, process normally
        image_urls = list_images_in_s3_folder(request.s3_folder_url)
        if not image_urls:
            return QRProcessResponse(
                success=False,
                message="No images found in folder",
                total_images=0,
                processed_images=0,
                results=[],
                error="No images found"
            )
        
        # Create task ID and folder record
        task_id = str(uuid.uuid4())
        folder_name = generate_folder_name_from_s3_folder(request.s3_folder_url)
        
        # Create database record (if database is available)
        if supabase:
            create_folder_record(task_id, folder_name, request.s3_folder_url, len(image_urls))
        
        # Process each image
        results = []
        processed_count = 0
        successful_count = 0
        total_qr_codes = 0
        
        for image_url in image_urls:
            result = await process_single_image(image_url)
            results.append(result)
            processed_count += 1
            
            if result.success:
                successful_count += 1
                total_qr_codes += len(result.qr_results)
            
            # Store result in database (if available)
            if supabase:
                store_image_result(task_id, image_url, result)
        
        # Update folder record as completed (if database is available)
        if supabase:
            update_folder_record(
                task_id,
                status='completed',
                processed_images=processed_count,
                successful_images=successful_count,
                total_qr_codes=total_qr_codes,
                processing_completed_at=datetime.now().isoformat()
            )
        
        logger.info(f"✅ Processing completed: {successful_count}/{len(image_urls)} images successful, {total_qr_codes} QR codes found")
        
        return QRProcessResponse(
            success=True,
            message=f"Processed {successful_count}/{len(image_urls)} images, found {total_qr_codes} QR codes",
            total_images=len(image_urls),
            processed_images=successful_count,
            results=results
        )
        
    except Exception as e:
        logger.error(f"QR processing error: {str(e)}")
        return QRProcessResponse(
            success=False,
            message="QR processing failed",
            total_images=0,
            processed_images=0,
            results=[],
            error=str(e)
        )

# Management endpoints for cached results
@app.get("/cached-folders")
async def list_cached_folders():
    """List all cached folder processing results"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        result = supabase.table('qr_processed_folders').select('*').order('created_at', desc=True).execute()
        return {
            "folders": result.data,
            "total_folders": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cached folders: {str(e)}")

@app.get("/cached-folders/{task_id}")
async def get_cached_folder_details(task_id: str):
    """Get detailed results for a specific cached folder"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get folder info
        folder_result = supabase.table('qr_processed_folders').select('*').eq('task_id', task_id).execute()
        if not folder_result.data:
            raise HTTPException(status_code=404, detail=f"Cached folder with task_id {task_id} not found")
        
        # Get image results
        images_result = supabase.table('qr_processed_images').select('*').eq('task_id', task_id).order('filename').execute()
        
        # Convert to API format
        results = [convert_cached_result_to_image_qr_result(result) for result in images_result.data]
        
        return {
            "folder_info": folder_result.data[0],
            "results": results,
            "total_images": len(results),
            "total_qr_codes": sum(len(result.qr_results) for result in results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cached folder details: {str(e)}")

@app.delete("/cached-folders/{task_id}")
async def delete_cached_folder(task_id: str):
    """Delete cached results for a specific folder"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Delete images first (foreign key constraint)
        supabase.table('qr_processed_images').delete().eq('task_id', task_id).execute()
        
        # Delete folder record
        result = supabase.table('qr_processed_folders').delete().eq('task_id', task_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Cached folder with task_id {task_id} not found")
        
        return {"message": f"Cached folder {task_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting cached folder: {str(e)}")

@app.post("/reprocess-folder")
async def reprocess_folder(request: QRProcessRequest):
    """Force reprocess a folder (ignore cache)"""
    if not supabase:
        # If no database, just process normally
        return await process_qr(request)
    
    try:
        logger.info(f"Force reprocessing folder: {request.s3_folder_url}")
        
        # Delete existing cache if it exists
        cached_folder = check_folder_in_database(request.s3_folder_url)
        if cached_folder:
            logger.info(f"Deleting existing cache for folder: {request.s3_folder_url}")
            # Delete images first
            supabase.table('qr_processed_images').delete().eq('task_id', cached_folder['task_id']).execute()
            # Delete folder record
            supabase.table('qr_processed_folders').delete().eq('task_id', cached_folder['task_id']).execute()
        
        # Process normally (will create new cache)
        return await process_qr(request)
        
    except Exception as e:
        logger.error(f"Error reprocessing folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reprocessing folder: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
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
import subprocess
import json
import math
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import numpy as np
from pyzbar import pyzbar
import requests
import tempfile
import uuid
from datetime import datetime
import qrcode
from io import BytesIO
import openai
from collections import defaultdict
import math
import exifread
from pyproj import Geod

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

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found - AI VIN description features will be disabled")
    openai_client = None
else:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")

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
# Path to AprilTag detector binary (tagStandard52h13 capable)
AT_EXE = "/home/ubuntu/workflows/qr_code_app/atagjs_example"


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
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
    bbox_visualization_url: Optional[str] = None  # Signed S3 URL for bounding box visualization

class QRProcessResponse(BaseModel):
    success: bool
    message: str
    total_images: int
    processed_images: int
    results: List[ImageQRResult] = []
    error: Optional[str] = None

# QR Generation Models
class QRGenerateRequest(BaseModel):
    description: Optional[str] = None
    size: int = 20  # Ultra-high resolution default for professional A4 printing

class QRCodeRecord(BaseModel):
    id: str
    qr_code_id: str  # The ID encoded in the QR code
    vin: Optional[str]  # Can be None if not assigned
    description: Optional[str]
    ai_description: Optional[str]
    s3_url: str
    image_url: str  # Signed URL
    created_at: datetime
    assigned_at: Optional[datetime]  # When VIN was assigned
    size: int
    is_active: bool

class QRGenerateResponse(BaseModel):
    success: bool
    qr_code: QRCodeRecord
    message: str

class QRListResponse(BaseModel):
    qr_codes: List[QRCodeRecord]
    total: int

# QR Reassignment Models
class QRReassignRequest(BaseModel):
    qr_code_id: str
    vin: str

class QRReassignResponse(BaseModel):
    success: bool
    qr_code: QRCodeRecord
    message: str

# Combined Generate and Assign Models
class QRGenerateAndAssignRequest(BaseModel):
    vin: str
    description: Optional[str] = None
    size: int = 20

class QRGenerateAndAssignResponse(BaseModel):
    success: bool
    qr_code: QRCodeRecord
    message: str

# VIN Tracking Models
class VINHistory(BaseModel):
    vin: str
    folder_name: str
    folder_s3_url: str
    image_filename: str
    image_url: str
    spotted_at: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float
    bbox_visualization_url: Optional[str] = None

class VINSummary(BaseModel):
    vin: str
    total_spottings: int
    first_spotted: datetime
    last_spotted: datetime
    folders: List[str]
    latest_location: Optional[Dict[str, float]] = None
    ai_description: Optional[str] = None

class VINListResponse(BaseModel):
    vins: List[VINSummary]
    total_vins: int

class VINHistoryResponse(BaseModel):
    vin: str
    history: List[VINHistory]
    total_spottings: int

class MapPoint(BaseModel):
    vin: str
    latitude: float
    longitude: float
    image_url: str
    spotted_at: datetime
    confidence: float

class MapData(BaseModel):
    folder_name: str
    folder_s3_url: str
    total_images: int
    total_vins: int
    points: List[MapPoint]
    bounds: Optional[Dict[str, float]] = None

class VINDashboardResponse(BaseModel):
    vins: List[VINSummary]
    latest_map: Optional[MapData]
    total_vins: int
    total_folders: int

# AprilTag Detection Models
class AprilTagDetection(BaseModel):
    id: int
    center: Dict[str, float]  # {"x": float, "y": float}
    corners: List[Dict[str, float]]  # [{"x": float, "y": float}, ...]
    confidence: float

class AprilTagDetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    format: str = "json"  # "json" or "binary"

class AprilTagDetectionResponse(BaseModel):
    success: bool
    detections: List[AprilTagDetection]
    total_detections: int
    message: str
    binary_data: Optional[bytes] = None  # For binary format output

class MovementPoint(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime
    confidence: float
    folder_name: str
    image_filename: str
    image_url: str

class VINMovementPath(BaseModel):
    vin: str
    total_points: int
    movement_points: List[MovementPoint]
    bounds: Optional[Dict[str, float]] = None
    total_distance_meters: Optional[float] = None

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

def estimate_object_gps(
    center_lat: float,
    center_lon: float,
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
    bbox_center_px: tuple[int, int],
    image_size_px: tuple[int, int]
):
    """
    Estimate object GPS coordinates from a single aerial image and its EXIF metadata.

    Args:
        center_lat, center_lon: image GPS center
        altitude_m: drone altitude in meters
        focal_length_mm: camera focal length (mm)
        sensor_width_mm: camera sensor width (mm)
        image_width_px: image width (px)
        bbox_center_px: (x, y) pixel coordinate of detected object
        image_size_px: (width, height) of image (px)

    Returns:
        (lat, lon): estimated GPS coordinate of object
    """

    # Ground width visible in image at current altitude
    ground_width_m = altitude_m * (sensor_width_mm / focal_length_mm)
    meters_per_px = ground_width_m / image_width_px

    # Pixel offset from center (x → east, y → south)
    dx_px = bbox_center_px[0] - (image_size_px[0] / 2)
    dy_px = bbox_center_px[1] - (image_size_px[1] / 2)
    dx_m = dx_px * meters_per_px
    dy_m = dy_px * meters_per_px

    # Convert local offsets to lat/lon
    geod = Geod(ellps="WGS84")
    lon_east, lat_east, _ = geod.fwd(center_lon, center_lat, 90, dx_m)
    lon_final, lat_final, _ = geod.fwd(lon_east, lat_east, 180, dy_m)

    return lat_final, lon_final


def convert_to_degrees(value):
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        d, m, s = value
        return float(d) + float(m)/60 + float(s)/3600
    return 0.0

def detect_apriltags_enhanced(image_data: bytes) -> List[Dict[str, Any]]:
    """Enhanced AprilTag detection with preprocessing. Returns list of detection dictionaries."""
    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply conservative image preprocessing
        import cv2
        import numpy as np
        
        img_array = np.array(img)
        
        # Apply gentle contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16,16))
        img_enhanced = clahe.apply(img_array)
        
        # Apply very light Gaussian blur to reduce noise
        img_denoised = cv2.GaussianBlur(img_enhanced, (1, 1), 0)
        
        # Convert back to PIL Image
        img_processed = Image.fromarray(img_denoised)
        
        # Save to temporary PGM file
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as f:
            tmp_path = f.name
        try:
            img_processed.save(tmp_path)
            
            # Run AprilTag detection
            result = subprocess.run([AT_EXE, tmp_path], 
                                  capture_output=True, text=True, check=True)
            
            # Parse JSON output
            json_text = '[]'
            for line in result.stdout.splitlines():
                idx = line.find('[')
                if idx != -1:
                    json_text = line[idx:]
                    break
            
            detections = json.loads(json_text)
            return detections
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not detect AprilTags: {str(e)}")
        return []

def decode_qr_from_base64(base64_string: str) -> List[str]:
    """Decode AprilTag (tagStandard52h13) from a base64 image using enhanced detection.
    Returns a list of all detected tag IDs."""
    try:
        # Strip data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Use enhanced detection
        detections = detect_apriltags_enhanced(image_data)
        
        if not detections:
            return []

        # Return all tag IDs as strings
        return [str(det.get("id")) for det in detections if det.get("id") is not None]
    except Exception as e:
        logger.warning(f"Could not decode AprilTag: {str(e)}")
        return []

# AI VIN Description Generation
def generate_vin_description(vin: str) -> str:
    """Generate AI-powered VIN description using ChatGPT"""
    if not openai_client:
        logger.warning("OpenAI client not available, skipping VIN description generation")
        return None
    
    try:
        prompt = f"""
        Analyze this VIN number and provide detailed vehicle information in the following format:
        VIN: {vin}
        
        Please provide ONLY the vehicle details in this exact format (no VIN prefix, no "Based on" text):
        - Make: [Manufacturer]
        - Model: [Model name]
        - Year: [Year]
        - Body: [Body style]
        - Engine: [Engine specification]
        - Assembly Plant: [Assembly location]
        
        If any information cannot be determined from the VIN, please indicate "Unknown" for that field.
        Do not include the VIN number or "Based on" text in your response.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a vehicle information expert. Analyze VIN numbers and provide accurate vehicle details."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        description = response.choices[0].message.content.strip()
        logger.info(f"Generated VIN description for {vin}")
        return description
        
    except Exception as e:
        logger.error(f"Error generating VIN description for {vin}: {e}")
        return None

# QR (now AprilTag) Generation Helper Functions
def generate_qr_code(data: str, size: int = 10) -> bytes:
    """Generate an AprilTag (tagStandard52h13) image as PNG bytes without OpenCV.
    Keeps the function name for compatibility with existing callers."""
    try:
        # Import locally to avoid hard dependency if not installed at import time
        from moms_apriltag import TagGenerator3
        
        # Use numeric sequence ID directly when possible; if not numeric, derive a stable small int
        try:
            tag_id = int(data)
        except ValueError:
            tag_id = abs(hash(str(data))) % 10000

        generator = TagGenerator3("tagStandard52h13")
        tag_img = generator.generate(tag_id)  # Returns a numpy array (grayscale)

        # Scale up for print quality (approx A4 height at 300 DPI)
        from PIL import Image as _PIL_Image
        pil_img = _PIL_Image.fromarray(tag_img)
        base_dpi = 300
        a4_height_inches = 11.69
        target_side = max(512, int(base_dpi * a4_height_inches))
        pil_img = pil_img.resize((target_side, target_side), resample=_PIL_Image.NEAREST)
        pil_img = pil_img.convert('RGB')

        buf = BytesIO()
        pil_img.save(buf, format='PNG', optimize=False, compress_level=0, dpi=(300, 300))
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error generating AprilTag (tagStandard52h13): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate AprilTag: {str(e)}")

def create_s3_key_for_qr(qr_code_id: str) -> str:
    """Create S3 key for AprilTag image (keeps name for compatibility)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = qr_code_id.replace("/", "_").replace("\\", "_")
    return f"apriltags/{safe_id}_{timestamp}.png"

def upload_qr_to_s3(file_bytes: bytes, s3_key: str) -> str:
    """Upload QR code file to S3 and return the S3 URL"""
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=os.getenv("S3_BUCKET", "rcsstoragebucket"),
            Key=s3_key,
            Body=file_bytes,
            ContentType='image/png'
        )
        s3_url = f"s3://{os.getenv('S3_BUCKET', 'rcsstoragebucket')}/{s3_key}"
        logger.info(f"Uploaded QR code to S3: {s3_url}")
        return s3_url
    except Exception as e:
        logger.error(f"Error uploading QR to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload QR to S3: {str(e)}")

def upload_bbox_visualization_to_s3(bbox_image_data, image_name: str, task_id: str) -> str:
    """Upload bounding box visualization to S3 and return the S3 URL"""
    try:
        # Handle different data types for bounding box visualization
        if isinstance(bbox_image_data, str):
            # If it's a string, try to decode as base64
            import base64
            try:
                image_data = base64.b64decode(bbox_image_data)
            except Exception as e:
                logger.warning(f"Failed to decode as base64, treating as raw data: {e}")
                image_data = bbox_image_data.encode('utf-8')
        elif isinstance(bbox_image_data, list):
            # If it's a list (raw image data), convert to bytes
            image_data = bytes(bbox_image_data)
        else:
            # Convert to bytes
            image_data = bytes(bbox_image_data)
        
        # Create S3 key for bounding box visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = image_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        s3_key = f"bbox_visualizations/{task_id}/{safe_filename}_{timestamp}_bbox.png"
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=os.getenv("S3_BUCKET", "rcsstoragebucket"),
            Key=s3_key,
            Body=image_data,
            ContentType='image/png'
        )
        
        s3_url = f"s3://{os.getenv('S3_BUCKET', 'rcsstoragebucket')}/{s3_key}"
        logger.info(f"✅ Uploaded bounding box visualization to: {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"Error uploading bounding box visualization to S3: {str(e)}")
        raise

def store_qr_record(qr_code_id: str, description: Optional[str], s3_url: str, size: int) -> str:
    """Store QR code record in database with ID-based system"""
    if not supabase:
        logger.warning("Supabase not available, skipping database storage")
        return str(uuid.uuid4())
    
    try:
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "qr_code_id": qr_code_id,
            "vin": None,  # VIN will be assigned later
            "description": description,
            "ai_description": None,  # Will be generated when VIN is assigned
            "s3_url": s3_url,
            "size": size,
            "created_at": datetime.now().isoformat(),
            "assigned_at": None,
            "is_active": True
        }
        
        result = supabase.table("qr_codes").insert(record).execute()
        logger.info(f"Stored QR record in database: {record_id} with QR code ID: {qr_code_id}")
        return record_id
    except Exception as e:
        logger.error(f"Error storing QR record: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store QR record: {str(e)}")

def get_qr_records(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Get QR code records from database"""
    if not supabase:
        logger.warning("Supabase not available, returning empty list")
        return []
    
    try:
        result = supabase.table("qr_codes").select("*").order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        return result.data
    except Exception as e:
        logger.error(f"Error fetching QR records: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch QR records: {str(e)}")

# VIN Tracking Helper Functions
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters using Haversine formula"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def cluster_vin_sightings(vin_data: List[Dict[str, Any]], radius_meters: float = 10.0) -> List[Dict[str, Any]]:
    """Cluster VIN sightings within a radius to count as single sightings"""
    if not vin_data:
        return []
    
    # Group by VIN
    vin_groups = defaultdict(list)
    for sighting in vin_data:
        vin_groups[sighting["vin"]].append(sighting)
    
    clustered_data = []
    
    for vin, sightings in vin_groups.items():
        if not sightings:
            continue
            
        # Sort by timestamp
        sightings.sort(key=lambda x: x["spotted_at"])
        
        # Cluster sightings within radius
        clusters = []
        for sighting in sightings:
            if not sighting.get("latitude") or not sighting.get("longitude"):
                # If no GPS data, treat as separate sighting
                clusters.append([sighting])
                continue
                
            # Check if this sighting belongs to an existing cluster
            added_to_cluster = False
            for cluster in clusters:
                # Check distance to any sighting in the cluster
                for cluster_sighting in cluster:
                    if (cluster_sighting.get("latitude") and cluster_sighting.get("longitude")):
                        distance = calculate_distance(
                            sighting["latitude"], sighting["longitude"],
                            cluster_sighting["latitude"], cluster_sighting["longitude"]
                        )
                        if distance <= radius_meters:
                            cluster.append(sighting)
                            added_to_cluster = True
                            break
                if added_to_cluster:
                    break
            
            if not added_to_cluster:
                clusters.append([sighting])
        
        # For each cluster, create a representative sighting
        for cluster in clusters:
            if not cluster:
                continue
                
            # Use the first sighting as representative
            representative = cluster[0].copy()
            
            # Update with cluster information
            representative["cluster_size"] = len(cluster)
            representative["total_confidence"] = sum(s.get("confidence", 0) for s in cluster)
            representative["avg_confidence"] = representative["total_confidence"] / len(cluster)
            
            # Use the most recent timestamp from the cluster
            representative["spotted_at"] = max(s["spotted_at"] for s in cluster)
            
            # Calculate average latitude and longitude for the cluster
            valid_coords = [s for s in cluster if s.get("latitude") is not None and s.get("longitude") is not None]
            if valid_coords:
                avg_lat = sum(s["latitude"] for s in valid_coords) / len(valid_coords)
                avg_lon = sum(s["longitude"] for s in valid_coords) / len(valid_coords)
                representative["latitude"] = avg_lat
                representative["longitude"] = avg_lon
                logger.info(f"Clustered {len(cluster)} sightings: avg lat={avg_lat:.6f}, avg lon={avg_lon:.6f}")
            else:
                logger.warning(f"No valid coordinates found in cluster of {len(cluster)} sightings")
            
            clustered_data.append(representative)
    
    return clustered_data

def get_all_vins_from_processed_images(folder_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get all VINs extracted from processed images with their history
    
    Args:
        folder_filter: Optional list of folder names to filter by
    """
    if not supabase:
        logger.warning("Supabase not available, returning empty list")
        return []
    
    try:
        # Get all processed images with QR results
        result = supabase.table("qr_processed_images").select(
            "id, filename, s3_input_url, image_signed_url, timestamp, latitude, longitude, qr_results, task_id, bbox_visualization_url"
        ).eq("processing_status", "success").execute()
        
        # Get folder information (always get all folders for proper mapping)
        folder_result = supabase.table("qr_processed_folders").select("task_id, folder_name, s3_input_folder_url").execute()
        folder_map = {folder["task_id"]: folder for folder in folder_result.data}
        
        # Extract VINs from QR results
        vin_data = []
        for image in result.data:
            if image.get("qr_results"):
                folder_info = folder_map.get(image["task_id"], {})
                
                # Generate fresh signed URL for each image
                fresh_image_url = get_s3_signed_url(image["s3_input_url"])
                
                for qr in image["qr_results"]:
                    content = qr.get("content", "")
                    # Resolve QR code content to VIN (handles both direct VINs and QR code IDs)
                    resolved_vin = resolve_qr_code_to_vin(content)
                    
                    if resolved_vin:
                        # Generate fresh signed URL for bounding box visualization if available
                        bbox_visualization_url = None
                        if image.get("bbox_visualization_url"):
                            bbox_url = image["bbox_visualization_url"]
                            try:
                                # If it's already a signed URL, extract the S3 key and generate fresh signed URL
                                if bbox_url.startswith("https://"):
                                    # Extract S3 key from signed URL
                                    # Format: https://bucket.s3.region.amazonaws.com/key?params
                                    from urllib.parse import urlparse
                                    parsed_url = urlparse(bbox_url)
                                    s3_key = parsed_url.path.lstrip('/')
                                    s3_uri = f"s3://{os.getenv('S3_BUCKET', 'rcsstoragebucket')}/{s3_key}"
                                    bbox_visualization_url = get_s3_signed_url(s3_uri)
                                else:
                                    # It's already an S3 URI, generate signed URL directly
                                    bbox_visualization_url = get_s3_signed_url(bbox_url)
                            except Exception as e:
                                logger.warning(f"Failed to generate fresh signed URL for bbox visualization: {str(e)}")
                        
                        vin_data.append({
                            "vin": resolved_vin,
                            "qr_code_id": content if len(content) != 17 else None,  # Store original QR code ID if not a direct VIN
                            "confidence": qr.get("confidence", 0.0),
                            "image_filename": image["filename"],
                            "image_url": fresh_image_url,
                            "spotted_at": image["timestamp"],
                            "latitude": image.get("latitude"),
                            "longitude": image.get("longitude"),
                            "folder_name": folder_info.get("folder_name", "Unknown"),
                            "folder_s3_url": folder_info.get("s3_input_folder_url", ""),
                            "task_id": image["task_id"],
                            "bbox_visualization_url": bbox_visualization_url
                        })
        
        # Apply folder filtering if specified
        if folder_filter:
            vin_data = [v for v in vin_data if v["folder_name"] in folder_filter]
        
        return vin_data
    except Exception as e:
        logger.error(f"Error fetching VIN data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch VIN data: {str(e)}")

def get_vin_summary(vin: str) -> VINSummary:
    """Get summary information for a specific VIN"""
    vin_data = get_all_vins_from_processed_images()
    vin_occurrences = [v for v in vin_data if v["vin"] == vin]
    
    if not vin_occurrences:
        raise HTTPException(status_code=404, detail=f"VIN {vin} not found")
    
    # Sort by timestamp
    vin_occurrences.sort(key=lambda x: x["spotted_at"])
    
    # Get unique folders
    folders = list(set([v["folder_name"] for v in vin_occurrences]))
    
    # Get latest location
    latest_location = None
    latest_occurrence = vin_occurrences[-1]
    if latest_occurrence.get("latitude") and latest_occurrence.get("longitude"):
        latest_location = {
            "latitude": latest_occurrence["latitude"],
            "longitude": latest_occurrence["longitude"]
        }
    
    # Get AI description from qr_codes table
    ai_description = None
    try:
        if supabase:
            qr_result = supabase.table("qr_codes").select("ai_description").eq("vin", vin).execute()
            if qr_result.data:
                ai_description = qr_result.data[0].get("ai_description")
    except Exception as e:
        logger.warning(f"Could not fetch AI description for VIN {vin}: {e}")
    
    return VINSummary(
        vin=vin,
        total_spottings=len(vin_occurrences),
        first_spotted=datetime.fromisoformat(vin_occurrences[0]["spotted_at"].replace('Z', '+00:00')),
        last_spotted=datetime.fromisoformat(vin_occurrences[-1]["spotted_at"].replace('Z', '+00:00')),
        folders=folders,
        latest_location=latest_location,
        ai_description=ai_description
    )

def get_latest_folder_map(folder_filter: Optional[List[str]] = None) -> Optional[MapData]:
    """Get map data for the latest processed folder
    
    Args:
        folder_filter: Optional list of folder names to filter by
    """
    if not supabase:
        return None
    
    try:
        # Get all completed folders and sort by folder name (which contains timestamp)
        if folder_filter:
            folder_result = supabase.table("qr_processed_folders").select(
                "task_id, folder_name, s3_input_folder_url, total_images, created_at"
            ).eq("status", "completed").in_("folder_name", folder_filter).execute()
        else:
            folder_result = supabase.table("qr_processed_folders").select(
                "task_id, folder_name, s3_input_folder_url, total_images, created_at"
            ).eq("status", "completed").execute()
        
        if not folder_result.data:
            return None
        
        # Sort by folder name datetime to get the most recent drone run chronologically
        def extract_datetime_from_folder_name(folder_record):
            """Extract datetime from folder name like 'Run September 29 5:33 PM'"""
            try:
                import re
                from datetime import datetime
                
                folder_name = folder_record["folder_name"]
                # Parse folder name format: "Run September 29 5:33 PM"
                match = re.match(r"Run (\w+) (\d+) (\d+):(\d+) (AM|PM)", folder_name)
                if match:
                    month_name, day, hour, minute, ampm = match.groups()
                    
                    # Convert month name to number
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    month = month_map.get(month_name, 1)
                    
                    # Convert 12-hour to 24-hour format
                    hour = int(hour)
                    if ampm == 'PM' and hour != 12:
                        hour += 12
                    elif ampm == 'AM' and hour == 12:
                        hour = 0
                    
                    # Create datetime object (assuming current year)
                    current_year = datetime.now().year
                    return datetime(current_year, month, int(day), hour, int(minute))
                
                # Fallback to created_at if parsing fails
                return datetime.fromisoformat(folder_record.get("created_at", "1970-01-01T00:00:00").replace('Z', '+00:00'))
            except:
                # Fallback to created_at if parsing fails
                return datetime.fromisoformat(folder_record.get("created_at", "1970-01-01T00:00:00").replace('Z', '+00:00'))
        
        # Sort by extracted datetime from folder name (most recent first)
        folders = sorted(folder_result.data, key=extract_datetime_from_folder_name, reverse=True)
        latest_folder = folders[0]
        
        # Get all images from this folder with VINs
        images_result = supabase.table("qr_processed_images").select(
            "filename, s3_input_url, image_signed_url, timestamp, latitude, longitude, qr_results"
        ).eq("task_id", latest_folder["task_id"]).eq("processing_status", "success").execute()
        
        # Collect all VIN points first
        all_points = []
        for image in images_result.data:
            if image.get("qr_results") and image.get("latitude") and image.get("longitude"):
                # Generate fresh signed URL for each image
                fresh_image_url = get_s3_signed_url(image["s3_input_url"])
                
                for qr in image["qr_results"]:
                    content = qr.get("content", "")
                    # Resolve QR code content to VIN (handles both direct VINs and QR code IDs)
                    resolved_vin = resolve_qr_code_to_vin(content)
                    
                    if resolved_vin:
                        all_points.append({
                            "vin": resolved_vin,
                            "qr_code_id": content if len(content) != 17 else None,  # Store original QR code ID if not a direct VIN
                            "latitude": float(image["latitude"]),
                            "longitude": float(image["longitude"]),
                            "image_url": fresh_image_url,
                            "spotted_at": datetime.fromisoformat(image["timestamp"].replace('Z', '+00:00')),
                            "confidence": qr.get("confidence", 0.0)
                        })
        
        # Apply clustering to group nearby VINs within this folder only
        clustered_points = cluster_vin_sightings(all_points, radius_meters=10.0)
        
        # Convert to MapPoint objects
        points = []
        for point in clustered_points:
            points.append(MapPoint(
                vin=point["vin"],
                latitude=point["latitude"],
                longitude=point["longitude"],
                image_url=point["image_url"],
                spotted_at=point["spotted_at"],
                confidence=point.get("avg_confidence", point.get("confidence", 0.0))
            ))
        
        # Calculate bounds
        bounds = None
        if points:
            lats = [p.latitude for p in points]
            lons = [p.longitude for p in points]
            bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons)
            }
        
        return MapData(
            folder_name=latest_folder["folder_name"],
            folder_s3_url=latest_folder["s3_input_folder_url"],
            total_images=latest_folder["total_images"],
            total_vins=len(set(p.vin for p in points)),
            points=points,
            bounds=bounds
        )
    except Exception as e:
        logger.error(f"Error fetching latest folder map: {e}")
        return None

def get_vin_movement_path(vin: str, folder_filter: Optional[List[str]] = None) -> VINMovementPath:
    """Get movement path for a specific VIN across all folders
    
    Args:
        vin: The VIN to get movement path for
        folder_filter: Optional list of folder names to filter by
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Get all processed images with QR results
        result = supabase.table("qr_processed_images").select(
            "id, filename, s3_input_url, image_signed_url, timestamp, latitude, longitude, qr_results, task_id"
        ).eq("processing_status", "success").execute()
        
        # Get folder information (always get all folders for proper mapping)
        folder_result = supabase.table("qr_processed_folders").select("task_id, folder_name").execute()
        folder_map = {folder["task_id"]: folder for folder in folder_result.data}
        
        # Extract movement points for this VIN
        movement_points = []
        for image in result.data:
            if image.get("qr_results") and image.get("latitude") and image.get("longitude"):
                folder_info = folder_map.get(image["task_id"], {})
                
                # Generate fresh signed URL for each image
                fresh_image_url = get_s3_signed_url(image["s3_input_url"])
                
                for qr in image["qr_results"]:
                    content = qr.get("content", "")
                    # Resolve QR code content to VIN and check if it matches the target VIN
                    resolved_vin = resolve_qr_code_to_vin(content)
                    if resolved_vin == vin:  # Match resolved VIN
                        movement_points.append(MovementPoint(
                            latitude=float(image["latitude"]),
                            longitude=float(image["longitude"]),
                            timestamp=datetime.fromisoformat(image["timestamp"].replace('Z', '+00:00')),
                            confidence=qr.get("confidence", 0.0),
                            folder_name=folder_info.get("folder_name", "Unknown"),
                            image_filename=image["filename"],
                            image_url=fresh_image_url
                        ))
        
        # Apply folder filtering if specified
        if folder_filter:
            movement_points = [p for p in movement_points if p.folder_name in folder_filter]
        
        # Sort by folder chronological order (drone flight sequence)
        # Extract time from folder name for proper chronological ordering
        def get_folder_timestamp(folder_name):
            try:
                # Parse "Run September 22 6:07 PM" format
                parts = folder_name.split()
                if len(parts) >= 4:
                    month = parts[1]
                    day = parts[2]
                    time_str = parts[3] + " " + parts[4]  # "6:07 PM"
                    
                    # Convert to sortable format
                    from datetime import datetime
                    # This is a simplified approach - in production you'd want more robust parsing
                    return folder_name  # For now, use string sorting which works for this format
                return folder_name
            except:
                return folder_name
        
        # Sort by folder name (which contains chronological timestamp)
        movement_points.sort(key=lambda x: get_folder_timestamp(x.folder_name))
        
        # Apply clustering within each folder (not across folders)
        clustered_movement_points = []
        current_folder = None
        current_folder_points = []
        
        for point in movement_points:
            if current_folder != point.folder_name:
                # Process previous folder's points
                if current_folder_points:
                    clustered_points = cluster_vin_sightings([
                        {
                            "vin": vin,
                            "latitude": p.latitude,
                            "longitude": p.longitude,
                            "spotted_at": p.timestamp.isoformat(),
                            "confidence": p.confidence,
                            "folder_name": p.folder_name,
                            "image_filename": p.image_filename,
                            "image_url": p.image_url
                        } for p in current_folder_points
                    ], radius_meters=15.0)
                    
                    # Convert back to MovementPoint objects
                    for clustered_point in clustered_points:
                        clustered_movement_points.append(MovementPoint(
                            latitude=clustered_point["latitude"],
                            longitude=clustered_point["longitude"],
                            timestamp=datetime.fromisoformat(clustered_point["spotted_at"].replace('Z', '+00:00')),
                            confidence=clustered_point.get("avg_confidence", clustered_point.get("confidence", 0.0)),
                            folder_name=clustered_point["folder_name"],
                            image_filename=clustered_point["image_filename"],
                            image_url=clustered_point["image_url"]
                        ))
                
                # Start new folder
                current_folder = point.folder_name
                current_folder_points = [point]
            else:
                current_folder_points.append(point)
        
        # Process the last folder
        if current_folder_points:
            clustered_points = cluster_vin_sightings([
                {
                    "vin": vin,
                    "latitude": p.latitude,
                    "longitude": p.longitude,
                    "spotted_at": p.timestamp.isoformat(),
                    "confidence": p.confidence,
                    "folder_name": p.folder_name,
                    "image_filename": p.image_filename,
                    "image_url": p.image_url
                } for p in current_folder_points
            ], radius_meters=15.0)
            
            # Convert back to MovementPoint objects
            for clustered_point in clustered_points:
                clustered_movement_points.append(MovementPoint(
                    latitude=clustered_point["latitude"],
                    longitude=clustered_point["longitude"],
                    timestamp=datetime.fromisoformat(clustered_point["spotted_at"].replace('Z', '+00:00')),
                    confidence=clustered_point.get("avg_confidence", clustered_point.get("confidence", 0.0)),
                    folder_name=clustered_point["folder_name"],
                    image_filename=clustered_point["image_filename"],
                    image_url=clustered_point["image_url"]
                ))
        
        movement_points = clustered_movement_points
        
        # Calculate bounds
        bounds = None
        if movement_points:
            lats = [p.latitude for p in movement_points]
            lons = [p.longitude for p in movement_points]
            bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons)
            }
        
        # Calculate total distance
        total_distance = 0.0
        if len(movement_points) > 1:
            for i in range(1, len(movement_points)):
                prev_point = movement_points[i-1]
                curr_point = movement_points[i]
                distance = calculate_distance(
                    prev_point.latitude, prev_point.longitude,
                    curr_point.latitude, curr_point.longitude
                )
                total_distance += distance
        
        return VINMovementPath(
            vin=vin,
            total_points=len(movement_points),
            movement_points=movement_points,
            bounds=bounds,
            total_distance_meters=total_distance if total_distance > 0 else None
        )
        
    except Exception as e:
        logger.error(f"Error fetching movement path for VIN {vin}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch movement path: {str(e)}")

def get_qr_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Get specific QR code record by database ID"""
    if not supabase:
        return None
    
    try:
        result = supabase.table("qr_codes").select("*").eq("id", record_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error fetching QR record {record_id}: {e}")
        return None

def get_qr_record_by_qr_code_id(qr_code_id: str) -> Optional[Dict[str, Any]]:
    """Get QR code record by QR code ID (the ID encoded in the QR)"""
    if not supabase:
        return None
    
    try:
        result = supabase.table("qr_codes").select("*").eq("qr_code_id", qr_code_id).eq("is_active", True).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error fetching QR record by QR code ID {qr_code_id}: {e}")
        return None

def assign_vin_to_qr_code(qr_code_id: str, vin: str) -> Optional[Dict[str, Any]]:
    """Assign a VIN to a QR code"""
    if not supabase:
        return None
    
    try:
        # Generate AI description for the VIN
        ai_description = generate_vin_description(vin)
        
        # Update the QR code record
        result = supabase.table("qr_codes").update({
            "vin": vin,
            "ai_description": ai_description,
            "assigned_at": datetime.now().isoformat()
        }).eq("qr_code_id", qr_code_id).eq("is_active", True).execute()
        
        if result.data:
            logger.info(f"Assigned VIN {vin} to QR code {qr_code_id}")
            return result.data[0]
        else:
            logger.warning(f"QR code {qr_code_id} not found or not active")
            return None
    except Exception as e:
        logger.error(f"Error assigning VIN {vin} to QR code {qr_code_id}: {e}")
        return None

def generate_unique_qr_code_id() -> str:
    """Generate a unique QR code ID using the DB sequence (numeric string).
    Falls back to a locally unique value if the database is unavailable."""
    # Prefer database-backed sequence to guarantee monotonic, non-repeating IDs
    if supabase:
        try:
            # Call Postgres function generate_qr_code_id() via RPC
            # This function should return the next sequence value as text
            rpc_result = supabase.rpc("generate_qr_code_id", {}).execute()

            # supabase-py returns the raw value in .data for scalar returns,
            # but can vary by version; normalize to string
            if rpc_result is not None and getattr(rpc_result, "data", None) is not None:
                value = rpc_result.data
                # value may be scalar, list, or dict depending on client version
                if isinstance(value, (int, float)):
                    return str(int(value))
                if isinstance(value, str):
                    return value
                if isinstance(value, list) and value:
                    # common pattern: [{"generate_qr_code_id": "123"}] or ["123"]
                    first = value[0]
                    if isinstance(first, dict):
                        # try common keys
                        for k in ("generate_qr_code_id", "qr_code_id", "id", "value"):
                            if k in first and first[k] is not None:
                                return str(first[k])
                    return str(first)
                if isinstance(value, dict):
                    for k in ("generate_qr_code_id", "qr_code_id", "id", "value"):
                        if k in value and value[k] is not None:
                            return str(value[k])
            # As a secondary attempt, select nextval directly if RPC isn't wired
            try:
                direct = supabase.rpc("sql", {"query": "select nextval('qr_code_id_seq')::text as id"}).execute()
                if direct is not None and getattr(direct, "data", None):
                    d = direct.data
                    if isinstance(d, list) and d and isinstance(d[0], dict) and "id" in d[0]:
                        return str(d[0]["id"])
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Falling back to local ID generation due to RPC error: {e}")

    # Fallback: generate a locally unique numeric-like string using timestamp
    # Note: This does not guarantee global monotonicity; used only if DB is unavailable
    from time import time
    return str(int(time() * 1000))

def resolve_qr_code_to_vin(qr_content: str) -> Optional[str]:
    """Resolve QR code content (ID) to VIN number"""
    if not qr_content:
        return None
    
    # Check if the content is already a VIN (17 characters, alphanumeric)
    if len(qr_content) == 17 and qr_content.isalnum():
        return qr_content
    
    # Otherwise, treat it as a QR code ID and look up the VIN
    if supabase:
        try:
            result = supabase.table("qr_codes").select("vin").eq("qr_code_id", qr_content).eq("is_active", True).execute()
            if result.data and result.data[0].get("vin"):
                return result.data[0]["vin"]
        except Exception as e:
            logger.warning(f"Error resolving QR code ID {qr_content} to VIN: {e}")
    
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
async def list_s3_folders(bucket: str = "rcsstoragebucket", prefix: str = "qr_sync/"):
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

def read_exif_basic(path, debug=False):
    """
    Simple function to extract all EXIF tags as listed by the user.
    """
    with open(path,'rb') as f:
        tags = exifread.process_file(f)
    
    def _deg(vals):
        d = float(vals[0].num)/vals[0].den
        m = float(vals[1].num)/vals[1].den
        s = float(vals[2].num)/vals[2].den
        return d + m/60.0 + s/3600.0
    
    # Basic GPS coordinates
    lat = lon = alt = heading = None
    if "GPS GPSLatitude" in tags:
        lat = _deg(tags["GPS GPSLatitude"].values)
        if tags.get("GPS GPSLatitudeRef", "N").printable != 'N': lat = -lat
    if "GPS GPSLongitude" in tags:
        lon = _deg(tags["GPS GPSLongitude"].values)
        if tags.get("GPS GPSLongitudeRef", "E").printable != 'E': lon = -lon
    if "GPS GPSAltitude" in tags:
        a = tags["GPS GPSAltitude"].values[0]
        alt = float(a.num)/float(a.den)
    if "GPS GPSImgDirection" in tags:
        d = tags["GPS GPSImgDirection"].values[0]
        heading = float(d.num)/float(d.den)
    
    # Extract all the tags as listed by the user
    realworld_data = {}
    
    # Basic file info
    realworld_data['filename'] = tags.get("EXIF FileName", {}).printable if "EXIF FileName" in tags and hasattr(tags["EXIF FileName"], 'printable') else None
    realworld_data['filesize'] = tags.get("EXIF FileSize", {}).printable if "EXIF FileSize" in tags and hasattr(tags["EXIF FileSize"], 'printable') else None
    realworld_data['filemodifydate'] = tags.get("EXIF FileModifyDate", {}).printable if "EXIF FileModifyDate" in tags and hasattr(tags["EXIF FileModifyDate"], 'printable') else None
    realworld_data['fileaccessdate'] = tags.get("EXIF FileAccessDate", {}).printable if "EXIF FileAccessDate" in tags and hasattr(tags["EXIF FileAccessDate"], 'printable') else None
    realworld_data['fileinodechangedate'] = tags.get("EXIF FileInodeChangeDate", {}).printable if "EXIF FileInodeChangeDate" in tags and hasattr(tags["EXIF FileInodeChangeDate"], 'printable') else None
    realworld_data['filepermissions'] = tags.get("EXIF FilePermissions", {}).printable if "EXIF FilePermissions" in tags and hasattr(tags["EXIF FilePermissions"], 'printable') else None
    realworld_data['filetype'] = tags.get("EXIF FileType", {}).printable if "EXIF FileType" in tags and hasattr(tags["EXIF FileType"], 'printable') else None
    realworld_data['filetypeextension'] = tags.get("EXIF FileTypeExtension", {}).printable if "EXIF FileTypeExtension" in tags and hasattr(tags["EXIF FileTypeExtension"], 'printable') else None
    realworld_data['mimetype'] = tags.get("EXIF MIMEType", {}).printable if "EXIF MIMEType" in tags and hasattr(tags["EXIF MIMEType"], 'printable') else None
    realworld_data['exifbyteorder'] = tags.get("EXIF ExifByteOrder", {}).printable if "EXIF ExifByteOrder" in tags and hasattr(tags["EXIF ExifByteOrder"], 'printable') else None
    realworld_data['imagewidth'] = tags.get("EXIF ImageWidth", {}).printable if "EXIF ImageWidth" in tags and hasattr(tags["EXIF ImageWidth"], 'printable') else None
    realworld_data['imageheight'] = tags.get("EXIF ImageHeight", {}).printable if "EXIF ImageHeight" in tags and hasattr(tags["EXIF ImageHeight"], 'printable') else None
    realworld_data['encodingprocess'] = tags.get("EXIF EncodingProcess", {}).printable if "EXIF EncodingProcess" in tags and hasattr(tags["EXIF EncodingProcess"], 'printable') else None
    realworld_data['bitspersample'] = tags.get("EXIF BitsPerSample", {}).printable if "EXIF BitsPerSample" in tags and hasattr(tags["EXIF BitsPerSample"], 'printable') else None
    realworld_data['colorcomponents'] = tags.get("EXIF ColorComponents", {}).printable if "EXIF ColorComponents" in tags and hasattr(tags["EXIF ColorComponents"], 'printable') else None
    realworld_data['ycbcrsubsampling'] = tags.get("EXIF YCbCrSubSampling", {}).printable if "EXIF YCbCrSubSampling" in tags and hasattr(tags["EXIF YCbCrSubSampling"], 'printable') else None
    realworld_data['imagedescription'] = tags.get("EXIF ImageDescription", {}).printable if "EXIF ImageDescription" in tags and hasattr(tags["EXIF ImageDescription"], 'printable') else None
    realworld_data['make'] = tags.get("Image Make", {}).printable if "Image Make" in tags and hasattr(tags["Image Make"], 'printable') else None
    realworld_data['model'] = tags.get("Image Model", {}).printable if "Image Model" in tags and hasattr(tags["Image Model"], 'printable') else None
    realworld_data['orientation'] = tags.get("Image Orientation", {}).printable if "Image Orientation" in tags and hasattr(tags["Image Orientation"], 'printable') else None
    realworld_data['xresolution'] = tags.get("EXIF XResolution", {}).printable if "EXIF XResolution" in tags and hasattr(tags["EXIF XResolution"], 'printable') else None
    realworld_data['yresolution'] = tags.get("EXIF YResolution", {}).printable if "EXIF YResolution" in tags and hasattr(tags["EXIF YResolution"], 'printable') else None
    realworld_data['resolutionunit'] = tags.get("EXIF ResolutionUnit", {}).printable if "EXIF ResolutionUnit" in tags and hasattr(tags["EXIF ResolutionUnit"], 'printable') else None
    realworld_data['software'] = tags.get("Image Software", {}).printable if "Image Software" in tags and hasattr(tags["Image Software"], 'printable') else None
    realworld_data['modifydate'] = tags.get("EXIF ModifyDate", {}).printable if "EXIF ModifyDate" in tags and hasattr(tags["EXIF ModifyDate"], 'printable') else None
    realworld_data['ycbcrpositioning'] = tags.get("EXIF YCbCrPositioning", {}).printable if "EXIF YCbCrPositioning" in tags and hasattr(tags["EXIF YCbCrPositioning"], 'printable') else None
    realworld_data['exposuretime'] = tags.get("EXIF ExposureTime", {}).printable if "EXIF ExposureTime" in tags and hasattr(tags["EXIF ExposureTime"], 'printable') else None
    realworld_data['fnumber'] = tags.get("EXIF FNumber", {}).printable if "EXIF FNumber" in tags and hasattr(tags["EXIF FNumber"], 'printable') else None
    realworld_data['exposureprogram'] = tags.get("EXIF ExposureProgram", {}).printable if "EXIF ExposureProgram" in tags and hasattr(tags["EXIF ExposureProgram"], 'printable') else None
    realworld_data['iso'] = tags.get("EXIF ISOSpeedRatings", {}).printable if "EXIF ISOSpeedRatings" in tags and hasattr(tags["EXIF ISOSpeedRatings"], 'printable') else None
    realworld_data['sensitivitytype'] = tags.get("EXIF SensitivityType", {}).printable if "EXIF SensitivityType" in tags and hasattr(tags["EXIF SensitivityType"], 'printable') else None
    realworld_data['exifversion'] = tags.get("EXIF ExifVersion", {}).printable if "EXIF ExifVersion" in tags and hasattr(tags["EXIF ExifVersion"], 'printable') else None
    realworld_data['datetimeoriginal'] = tags.get("EXIF DateTimeOriginal", {}).printable if "EXIF DateTimeOriginal" in tags and hasattr(tags["EXIF DateTimeOriginal"], 'printable') else None
    realworld_data['createdate'] = tags.get("EXIF CreateDate", {}).printable if "EXIF CreateDate" in tags and hasattr(tags["EXIF CreateDate"], 'printable') else None
    realworld_data['componentsconfiguration'] = tags.get("EXIF ComponentsConfiguration", {}).printable if "EXIF ComponentsConfiguration" in tags and hasattr(tags["EXIF ComponentsConfiguration"], 'printable') else None
    realworld_data['shutterspeedvalue'] = tags.get("EXIF ShutterSpeedValue", {}).printable if "EXIF ShutterSpeedValue" in tags and hasattr(tags["EXIF ShutterSpeedValue"], 'printable') else None
    realworld_data['aperturevalue'] = tags.get("EXIF ApertureValue", {}).printable if "EXIF ApertureValue" in tags and hasattr(tags["EXIF ApertureValue"], 'printable') else None
    realworld_data['exposurecompensation'] = tags.get("EXIF ExposureBiasValue", {}).printable if "EXIF ExposureBiasValue" in tags and hasattr(tags["EXIF ExposureBiasValue"], 'printable') else None
    realworld_data['maxaperturevalue'] = tags.get("EXIF MaxApertureValue", {}).printable if "EXIF MaxApertureValue" in tags and hasattr(tags["EXIF MaxApertureValue"], 'printable') else None
    realworld_data['subjectdistance'] = tags.get("EXIF SubjectDistance", {}).printable if "EXIF SubjectDistance" in tags and hasattr(tags["EXIF SubjectDistance"], 'printable') else None
    realworld_data['meteringmode'] = tags.get("EXIF MeteringMode", {}).printable if "EXIF MeteringMode" in tags and hasattr(tags["EXIF MeteringMode"], 'printable') else None
    realworld_data['lightsource'] = tags.get("EXIF LightSource", {}).printable if "EXIF LightSource" in tags and hasattr(tags["EXIF LightSource"], 'printable') else None
    realworld_data['flash'] = tags.get("EXIF Flash", {}).printable if "EXIF Flash" in tags and hasattr(tags["EXIF Flash"], 'printable') else None
    realworld_data['focallength'] = tags.get("EXIF FocalLength", {}).printable if "EXIF FocalLength" in tags and hasattr(tags["EXIF FocalLength"], 'printable') else None
    realworld_data['flashpixversion'] = tags.get("EXIF FlashpixVersion", {}).printable if "EXIF FlashpixVersion" in tags and hasattr(tags["EXIF FlashpixVersion"], 'printable') else None
    realworld_data['colorspace'] = tags.get("EXIF ColorSpace", {}).printable if "EXIF ColorSpace" in tags and hasattr(tags["EXIF ColorSpace"], 'printable') else None
    realworld_data['exifimagewidth'] = tags.get("EXIF ExifImageWidth", {}).printable if "EXIF ExifImageWidth" in tags and hasattr(tags["EXIF ExifImageWidth"], 'printable') else None
    realworld_data['exifimageheight'] = tags.get("EXIF ExifImageHeight", {}).printable if "EXIF ExifImageHeight" in tags and hasattr(tags["EXIF ExifImageHeight"], 'printable') else None
    realworld_data['interopindex'] = tags.get("EXIF InteropIndex", {}).printable if "EXIF InteropIndex" in tags and hasattr(tags["EXIF InteropIndex"], 'printable') else None
    realworld_data['interopversion'] = tags.get("EXIF InteropVersion", {}).printable if "EXIF InteropVersion" in tags and hasattr(tags["EXIF InteropVersion"], 'printable') else None
    realworld_data['filesource'] = tags.get("EXIF FileSource", {}).printable if "EXIF FileSource" in tags and hasattr(tags["EXIF FileSource"], 'printable') else None
    realworld_data['scenetype'] = tags.get("EXIF SceneType", {}).printable if "EXIF SceneType" in tags and hasattr(tags["EXIF SceneType"], 'printable') else None
    realworld_data['customrendered'] = tags.get("EXIF CustomRendered", {}).printable if "EXIF CustomRendered" in tags and hasattr(tags["EXIF CustomRendered"], 'printable') else None
    realworld_data['exposuremode'] = tags.get("EXIF ExposureMode", {}).printable if "EXIF ExposureMode" in tags and hasattr(tags["EXIF ExposureMode"], 'printable') else None
    realworld_data['whitebalance'] = tags.get("EXIF WhiteBalance", {}).printable if "EXIF WhiteBalance" in tags and hasattr(tags["EXIF WhiteBalance"], 'printable') else None
    realworld_data['digitalzoomratio'] = tags.get("EXIF DigitalZoomRatio", {}).printable if "EXIF DigitalZoomRatio" in tags and hasattr(tags["EXIF DigitalZoomRatio"], 'printable') else None
    realworld_data['focallengthin35mmformat'] = tags.get("EXIF FocalLengthIn35mmFilm", {}).printable if "EXIF FocalLengthIn35mmFilm" in tags and hasattr(tags["EXIF FocalLengthIn35mmFilm"], 'printable') else None
    realworld_data['scenecapturetype'] = tags.get("EXIF SceneCaptureType", {}).printable if "EXIF SceneCaptureType" in tags and hasattr(tags["EXIF SceneCaptureType"], 'printable') else None
    realworld_data['gaincontrol'] = tags.get("EXIF GainControl", {}).printable if "EXIF GainControl" in tags and hasattr(tags["EXIF GainControl"], 'printable') else None
    realworld_data['contrast'] = tags.get("EXIF Contrast", {}).printable if "EXIF Contrast" in tags and hasattr(tags["EXIF Contrast"], 'printable') else None
    realworld_data['saturation'] = tags.get("EXIF Saturation", {}).printable if "EXIF Saturation" in tags and hasattr(tags["EXIF Saturation"], 'printable') else None
    realworld_data['sharpness'] = tags.get("EXIF Sharpness", {}).printable if "EXIF Sharpness" in tags and hasattr(tags["EXIF Sharpness"], 'printable') else None
    realworld_data['devicesettingdescription'] = tags.get("EXIF DeviceSettingDescription", {}).printable if "EXIF DeviceSettingDescription" in tags and hasattr(tags["EXIF DeviceSettingDescription"], 'printable') else None
    realworld_data['serialnumber'] = tags.get("EXIF SerialNumber", {}).printable if "EXIF SerialNumber" in tags and hasattr(tags["EXIF SerialNumber"], 'printable') else None
    realworld_data['lensinfo'] = tags.get("EXIF LensInfo", {}).printable if "EXIF LensInfo" in tags and hasattr(tags["EXIF LensInfo"], 'printable') else None
    realworld_data['uniquecameramodel'] = tags.get("EXIF UniqueCameraModel", {}).printable if "EXIF UniqueCameraModel" in tags and hasattr(tags["EXIF UniqueCameraModel"], 'printable') else None
    realworld_data['gpsversionid'] = tags.get("GPS GPSVersionID", {}).printable if "GPS GPSVersionID" in tags and hasattr(tags["GPS GPSVersionID"], 'printable') else None
    realworld_data['gpslatituderef'] = tags.get("GPS GPSLatitudeRef", {}).printable if "GPS GPSLatitudeRef" in tags and hasattr(tags["GPS GPSLatitudeRef"], 'printable') else None
    realworld_data['gpslatitude'] = tags.get("GPS GPSLatitude", {}).printable if "GPS GPSLatitude" in tags and hasattr(tags["GPS GPSLatitude"], 'printable') else None
    realworld_data['gpslongituderef'] = tags.get("GPS GPSLongitudeRef", {}).printable if "GPS GPSLongitudeRef" in tags and hasattr(tags["GPS GPSLongitudeRef"], 'printable') else None
    realworld_data['gpslongitude'] = tags.get("GPS GPSLongitude", {}).printable if "GPS GPSLongitude" in tags and hasattr(tags["GPS GPSLongitude"], 'printable') else None
    realworld_data['gpsaltituderef'] = tags.get("GPS GPSAltitudeRef", {}).printable if "GPS GPSAltitudeRef" in tags and hasattr(tags["GPS GPSAltitudeRef"], 'printable') else None
    realworld_data['gpsaltitude'] = tags.get("GPS GPSAltitude", {}).printable if "GPS GPSAltitude" in tags and hasattr(tags["GPS GPSAltitude"], 'printable') else None
    realworld_data['gpsstatus'] = tags.get("GPS GPSStatus", {}).printable if "GPS GPSStatus" in tags and hasattr(tags["GPS GPSStatus"], 'printable') else None
    realworld_data['gpsmapdatum'] = tags.get("GPS GPSMapDatum", {}).printable if "GPS GPSMapDatum" in tags and hasattr(tags["GPS GPSMapDatum"], 'printable') else None
    realworld_data['xpcomment'] = tags.get("EXIF XPComment", {}).printable if "EXIF XPComment" in tags and hasattr(tags["EXIF XPComment"], 'printable') else None
    realworld_data['xpkeywords'] = tags.get("EXIF XPKeywords", {}).printable if "EXIF XPKeywords" in tags and hasattr(tags["EXIF XPKeywords"], 'printable') else None
    realworld_data['compression'] = tags.get("EXIF Compression", {}).printable if "EXIF Compression" in tags and hasattr(tags["EXIF Compression"], 'printable') else None
    realworld_data['thumbnailoffset'] = tags.get("EXIF ThumbnailOffset", {}).printable if "EXIF ThumbnailOffset" in tags and hasattr(tags["EXIF ThumbnailOffset"], 'printable') else None
    realworld_data['thumbnaillength'] = tags.get("EXIF ThumbnailLength", {}).printable if "EXIF ThumbnailLength" in tags and hasattr(tags["EXIF ThumbnailLength"], 'printable') else None
    realworld_data['about'] = tags.get("EXIF About", {}).printable if "EXIF About" in tags and hasattr(tags["EXIF About"], 'printable') else None
    realworld_data['format'] = tags.get("EXIF Format", {}).printable if "EXIF Format" in tags and hasattr(tags["EXIF Format"], 'printable') else None
    realworld_data['imagesource'] = tags.get("EXIF ImageSource", {}).printable if "EXIF ImageSource" in tags and hasattr(tags["EXIF ImageSource"], 'printable') else None
    realworld_data['gpsstatus_rtk'] = tags.get("EXIF GpsStatus", {}).printable if "EXIF GpsStatus" in tags and hasattr(tags["EXIF GpsStatus"], 'printable') else None
    realworld_data['altitudetype'] = tags.get("EXIF AltitudeType", {}).printable if "EXIF AltitudeType" in tags and hasattr(tags["EXIF AltitudeType"], 'printable') else None
    realworld_data['absolutealtitude'] = tags.get("EXIF AbsoluteAltitude", {}).printable if "EXIF AbsoluteAltitude" in tags and hasattr(tags["EXIF AbsoluteAltitude"], 'printable') else None
    realworld_data['relativealtitude'] = tags.get("EXIF RelativeAltitude", {}).printable if "EXIF RelativeAltitude" in tags and hasattr(tags["EXIF RelativeAltitude"], 'printable') else None
    realworld_data['gimbalrolldegree'] = tags.get("EXIF GimbalRollDegree", {}).printable if "EXIF GimbalRollDegree" in tags and hasattr(tags["EXIF GimbalRollDegree"], 'printable') else None
    realworld_data['gimbalyawdegree'] = tags.get("EXIF GimbalYawDegree", {}).printable if "EXIF GimbalYawDegree" in tags and hasattr(tags["EXIF GimbalYawDegree"], 'printable') else None
    realworld_data['gimbalpitchdegree'] = tags.get("EXIF GimbalPitchDegree", {}).printable if "EXIF GimbalPitchDegree" in tags and hasattr(tags["EXIF GimbalPitchDegree"], 'printable') else None
    realworld_data['flightrolldegree'] = tags.get("EXIF FlightRollDegree", {}).printable if "EXIF FlightRollDegree" in tags and hasattr(tags["EXIF FlightRollDegree"], 'printable') else None
    realworld_data['flightyawdegree'] = tags.get("EXIF FlightYawDegree", {}).printable if "EXIF FlightYawDegree" in tags and hasattr(tags["EXIF FlightYawDegree"], 'printable') else None
    realworld_data['flightpitchdegree'] = tags.get("EXIF FlightPitchDegree", {}).printable if "EXIF FlightPitchDegree" in tags and hasattr(tags["EXIF FlightPitchDegree"], 'printable') else None
    realworld_data['flightxspeed'] = tags.get("EXIF FlightXSpeed", {}).printable if "EXIF FlightXSpeed" in tags and hasattr(tags["EXIF FlightXSpeed"], 'printable') else None
    realworld_data['flightyspeed'] = tags.get("EXIF FlightYSpeed", {}).printable if "EXIF FlightYSpeed" in tags and hasattr(tags["EXIF FlightYSpeed"], 'printable') else None
    realworld_data['flightzspeed'] = tags.get("EXIF FlightZSpeed", {}).printable if "EXIF FlightZSpeed" in tags and hasattr(tags["EXIF FlightZSpeed"], 'printable') else None
    realworld_data['camreverse'] = tags.get("EXIF CamReverse", {}).printable if "EXIF CamReverse" in tags and hasattr(tags["EXIF CamReverse"], 'printable') else None
    realworld_data['gimbalreverse'] = tags.get("EXIF GimbalReverse", {}).printable if "EXIF GimbalReverse" in tags and hasattr(tags["EXIF GimbalReverse"], 'printable') else None
    realworld_data['sensortemperature'] = tags.get("EXIF SensorTemperature", {}).printable if "EXIF SensorTemperature" in tags and hasattr(tags["EXIF SensorTemperature"], 'printable') else None
    realworld_data['productname'] = tags.get("EXIF ProductName", {}).printable if "EXIF ProductName" in tags and hasattr(tags["EXIF ProductName"], 'printable') else None
    realworld_data['selfdata'] = tags.get("EXIF SelfData", {}).printable if "EXIF SelfData" in tags and hasattr(tags["EXIF SelfData"], 'printable') else None
    realworld_data['surveyingmode'] = tags.get("EXIF SurveyingMode", {}).printable if "EXIF SurveyingMode" in tags and hasattr(tags["EXIF SurveyingMode"], 'printable') else None
    realworld_data['shuttertype'] = tags.get("EXIF ShutterType", {}).printable if "EXIF ShutterType" in tags and hasattr(tags["EXIF ShutterType"], 'printable') else None
    realworld_data['cameraserialnumber'] = tags.get("EXIF CameraSerialNumber", {}).printable if "EXIF CameraSerialNumber" in tags and hasattr(tags["EXIF CameraSerialNumber"], 'printable') else None
    realworld_data['dronemodel'] = tags.get("EXIF DroneModel", {}).printable if "EXIF DroneModel" in tags and hasattr(tags["EXIF DroneModel"], 'printable') else None
    realworld_data['droneserialnumber'] = tags.get("EXIF DroneSerialNumber", {}).printable if "EXIF DroneSerialNumber" in tags and hasattr(tags["EXIF DroneSerialNumber"], 'printable') else None
    realworld_data['whitebalancecct'] = tags.get("EXIF WhiteBalanceCCT", {}).printable if "EXIF WhiteBalanceCCT" in tags and hasattr(tags["EXIF WhiteBalanceCCT"], 'printable') else None
    realworld_data['sensorfps'] = tags.get("EXIF SensorFPS", {}).printable if "EXIF SensorFPS" in tags and hasattr(tags["EXIF SensorFPS"], 'printable') else None
    realworld_data['version'] = tags.get("EXIF Version", {}).printable if "EXIF Version" in tags and hasattr(tags["EXIF Version"], 'printable') else None
    realworld_data['hassettings'] = tags.get("EXIF HasSettings", {}).printable if "EXIF HasSettings" in tags and hasattr(tags["EXIF HasSettings"], 'printable') else None
    realworld_data['hascrop'] = tags.get("EXIF HasCrop", {}).printable if "EXIF HasCrop" in tags and hasattr(tags["EXIF HasCrop"], 'printable') else None
    realworld_data['alreadyapplied'] = tags.get("EXIF AlreadyApplied", {}).printable if "EXIF AlreadyApplied" in tags and hasattr(tags["EXIF AlreadyApplied"], 'printable') else None
    realworld_data['totalframes'] = tags.get("EXIF TotalFrames", {}).printable if "EXIF TotalFrames" in tags and hasattr(tags["EXIF TotalFrames"], 'printable') else None
    realworld_data['sensorid'] = tags.get("EXIF SensorID", {}).printable if "EXIF SensorID" in tags and hasattr(tags["EXIF SensorID"], 'printable') else None
    realworld_data['scalefactor35efl'] = tags.get("EXIF ScaleFactor35efl", {}).printable if "EXIF ScaleFactor35efl" in tags and hasattr(tags["EXIF ScaleFactor35efl"], 'printable') else None
    realworld_data['shutterspeed'] = tags.get("EXIF ShutterSpeed", {}).printable if "EXIF ShutterSpeed" in tags and hasattr(tags["EXIF ShutterSpeed"], 'printable') else None
    realworld_data['circleofconfusion'] = tags.get("EXIF CircleOfConfusion", {}).printable if "EXIF CircleOfConfusion" in tags and hasattr(tags["EXIF CircleOfConfusion"], 'printable') else None
    realworld_data['fov'] = tags.get("EXIF FOV", {}).printable if "EXIF FOV" in tags and hasattr(tags["EXIF FOV"], 'printable') else None
    realworld_data['focallength35efl'] = tags.get("EXIF FocalLength35efl", {}).printable if "EXIF FocalLength35efl" in tags and hasattr(tags["EXIF FocalLength35efl"], 'printable') else None
    realworld_data['gpsposition'] = tags.get("EXIF GPSPosition", {}).printable if "EXIF GPSPosition" in tags and hasattr(tags["EXIF GPSPosition"], 'printable') else None
    realworld_data['hyperfocaldistance'] = tags.get("EXIF HyperfocalDistance", {}).printable if "EXIF HyperfocalDistance" in tags and hasattr(tags["EXIF HyperfocalDistance"], 'printable') else None
    realworld_data['lightvalue'] = tags.get("EXIF LightValue", {}).printable if "EXIF LightValue" in tags and hasattr(tags["EXIF LightValue"], 'printable') else None
    
    # Print the extracted data
    print("EXTRACTED EXIF DATA:")
    for key, value in realworld_data.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    return lat, lon, alt, heading, realworld_data


def pixel_to_latlon(
    x: float,
    y: float,
    W: float,
    H: float,
    W_m: float,
    H_m: float,
    lat_c: float,
    lon_c: float,
    gimbal_yaw_deg: float
):
    """
    Convert image pixel (x, y) to latitude/longitude using image center,
    ground coverage, and gimbal yaw.

    Parameters
    ----------
    x, y : float
        Pixel coordinates (origin top-left, x→right, y→down) //this will be the center of the april tag from the top left of the entrie photo.
    W, H : float
        Image width and height in pixels 
    W_m, H_m : float
        Ground coverage width and height in meters .
    lat_c, lon_c : float
        Latitude and longitude of the image center (decimal degrees).
    gimbal_yaw_deg : float
        Gimbal yaw (degrees). Positive = clockwise from true north.
        Example: -123.2 from your telemetry.

    Returns
    -------
    lat, lon : tuple(float, float)
        Latitude and longitude of the pixel (decimal degrees).
    """
    import math

    # --- Step 1: basic constants ---
    meters_per_px_x = W_m / W
    meters_per_px_y = H_m / H
    theta = math.radians(gimbal_yaw_deg)  # convert to radians

    # --- Step 2: pixel offset from image center (y positive up) ---
    x_img = (x - W / 2.0) * meters_per_px_x
    y_img = (H / 2.0 - y) * meters_per_px_y

    # --- Step 3: rotate image coordinates by yaw ---
    east_m  =  x_img * math.cos(theta) + y_img * math.sin(theta)
    north_m = -x_img * math.sin(theta) + y_img * math.cos(theta)

    # --- Step 4: convert meters → degrees ---
    dlat = north_m / 111320.0
    dlon = east_m / (111320.0 * math.cos(math.radians(lat_c)))

    # --- Step 5: final lat/lon ---
    lat = lat_c + dlat
    lon = lon_c + dlon
    return lat, lon

def read_xmp_metadata(image_path: str) -> Dict[str, str]:
    """Read XMP metadata from image file"""
    try:
        # Read the file as binary and look for XMP data
        with open(image_path, 'rb') as f:
            content = f.read()
            
        # Look for XMP data patterns
        xmp_data = {}
        
        # Look for orientation data in the file content
        orientation_patterns = [
            b'GimbalRollDegree',
            b'GimbalYawDegree', 
            b'GimbalPitchDegree',
            b'FlightRollDegree',
            b'FlightYawDegree',
            b'FlightPitchDegree'
        ]
        
        for pattern in orientation_patterns:
            # Find the pattern in the file
            pos = content.find(pattern)
            if pos != -1:
                # Look for the value after the pattern
                # Find the next '=' after the pattern
                value_start = pos + len(pattern)
                while value_start < len(content) and content[value_start] != b'='[0]:
                    value_start += 1
                
                if value_start < len(content):
                    # Skip the '=' and look for the opening quote
                    value_start += 1
                    while value_start < len(content) and content[value_start] != b'"'[0]:
                        value_start += 1
                    
                    if value_start < len(content):
                        # Skip the opening quote
                        value_start += 1
                        # Find the closing quote
                        value_end = value_start
                        while value_end < len(content) and content[value_end] != b'"'[0]:
                            value_end += 1
                        
                        if value_end > value_start:
                            try:
                                value = content[value_start:value_end].decode('utf-8', errors='ignore').strip()
                                # Clean up the value (remove + sign, etc.)
                                if value.startswith('+'):
                                    value = value[1:]
                                xmp_data[pattern.decode('utf-8').lower()] = value
                            except:
                                pass
        
        if xmp_data:
            print(f"DEBUG: Found XMP orientation data: {xmp_data}")
            return xmp_data
                
    except Exception as e:
        logger.error(f"Error reading XMP metadata: {str(e)}")
    
    return None


def get_realworld_coordinates(s3_image_url: str) -> List[Dict[str, float]]:
    """Get realworld coordinates from image coordinates"""
    try:
        # Download image from S3 locally for processing
        import os
        
        # Parse S3 URL
        if s3_image_url.startswith('s3://'):
            # Extract bucket and key from s3://bucket/key format
            s3_path = s3_image_url[5:]  # Remove 's3://'
            bucket, key = s3_path.split('/', 1)
            
            # Download from S3
            s3_client = boto3.client('s3')
            local_image_path = "temp_image.jpg"
            s3_client.download_file(bucket, key, local_image_path)
        else:
            # Handle regular HTTP URLs
            import requests
            response = requests.get(s3_image_url)
            local_image_path = "temp_image.jpg"
            with open(local_image_path, 'wb') as f:
                f.write(response.content)
        
        # Extract EXIF data
        lat, lon, alt, heading, realworld_data = read_exif_basic(local_image_path)
        
        # Also extract XMP metadata for orientation data
        xmp_data = read_xmp_metadata(local_image_path)
        if xmp_data:
            realworld_data.update(xmp_data)
        
        if lat is None or lon is None or alt is None:
            logger.error("Missing GPS data in EXIF")
            return None
        
        # Run enhanced AprilTag detection
        try:
            # Read image data
            with open(local_image_path, 'rb') as f:
                image_data = f.read()
            
            # Use enhanced detection with preprocessing
            detections = detect_apriltags_enhanced(image_data)
            logger.info(f"Enhanced AprilTag detector found {len(detections)} detections in image")
            
        
            if detections:
                # Process all detections
                results = []
                
                # Get image dimensions from the image itself (same for all detections)
                from PIL import Image
                with Image.open(local_image_path) as img:
                    image_width, image_height = img.size
                
                # Extract focal length from EXIF data (same for all detections)
                focal_length_mm = None
                if realworld_data.get('focallength'):
                    try:
                        # Handle fraction format like "168/25"
                        if '/' in str(realworld_data['focallength']):
                            num, den = str(realworld_data['focallength']).split('/')
                            focal_length_mm = float(num) / float(den)
                        else:
                            focal_length_mm = float(realworld_data['focallength'])
                    except:
                        pass
                
                if focal_length_mm is None:
                    logger.error("Could not extract focal length from EXIF")
                    return None
                
                # Image center coordinates (drone position) - same for all detections
                image_center_x = image_width / 2
                image_center_y = image_height / 2
                
                # Convert pixel offset to meters using ground sample distance
                # Use 35mm equivalent focal length for GSD calculation
                focal_length_35mm = None
                
                # Read EXIF tags to get 35mm equivalent focal length
                import exifread
                with open(local_image_path, 'rb') as f:
                    tags = exifread.process_file(f)
                
                if "EXIF FocalLengthIn35mmFilm" in tags:
                    focal_length_35mm_tag = tags["EXIF FocalLengthIn35mmFilm"]
                    if hasattr(focal_length_35mm_tag, 'printable'):
                        focal_length_35mm = float(focal_length_35mm_tag.printable)
                
                if focal_length_35mm is None:
                    focal_length_35mm = focal_length_mm  # fallback to actual focal length
                
                # Get digital zoom ratio and apply it to focal length
                digital_zoom_ratio = 1.0  # default no zoom
                if "EXIF DigitalZoomRatio" in tags:
                    zoom_tag = tags["EXIF DigitalZoomRatio"]
                    if hasattr(zoom_tag, 'printable'):
                        try:
                            if '/' in zoom_tag.printable:
                                num, den = map(float, zoom_tag.printable.split('/'))
                                digital_zoom_ratio = num / den
                            else:
                                digital_zoom_ratio = float(zoom_tag.printable)
                        except (ValueError, ZeroDivisionError):
                            digital_zoom_ratio = 1.0
                
                # Apply zoom to effective focal length
                effective_focal_length = focal_length_35mm * digital_zoom_ratio
                
                # Define sensor width for DJI M3TD with 1/2-inch CMOS sensor
                sensor_width_mm = 6.4  # 1/2-inch sensor width (4:3 aspect ratio)
                
                # Ground sample distance = (altitude * sensor_width) / (effective_focal_length * image_width)
                gsd = (alt * sensor_width_mm) / (effective_focal_length * image_width)

            
                
                # Calculate ground coverage in meters using GSD
                W_m = image_width * gsd  # Ground coverage width in meters
                H_m = image_height * gsd  # Ground coverage height in meters
                
                # Get gimbal yaw from XMP data
                gimbal_yaw_deg = 0.0
                if realworld_data.get('gimbalyawdegree'):
                    try:
                        gimbal_yaw_deg = float(realworld_data['gimbalyawdegree'])
                    except:
                        gimbal_yaw_deg = 0.0
                
                # Process each detection
                logger.info(f"Processing {len(detections)} AprilTag detections for coordinate calculation")
                for detection in detections:
                    # Get center coordinates from AprilTag detection
                    center = detection['center']
                    bbox_center_x = center['x']
                    bbox_center_y = center['y']
                    apriltag_id = detection.get('id', 'unknown')
                    logger.info(f"Processing AprilTag ID {apriltag_id} at pixel coordinates ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
                    
                    # Calculate offset from image center to AprilTag center
                    offset_x_px = bbox_center_x - image_center_x
                    offset_y_px = bbox_center_y - image_center_y
                    
                    print(f"DEBUG: Ground coverage: {W_m:.2f}m x {H_m:.2f}m")
                    print(f"DEBUG: Gimbal yaw: {gimbal_yaw_deg:.2f}°")
                    print(f"DEBUG: AprilTag center: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
                    
                    # Use the new pixel_to_latlon function
                    object_lat, object_lon = pixel_to_latlon(
                        bbox_center_x, bbox_center_y,  # AprilTag center coordinates
                        image_width, image_height,      # Image dimensions
                        W_m, H_m,                      # Ground coverage in meters
                        lat, lon,                      # Image center (drone position)
                        gimbal_yaw_deg                 # Gimbal yaw angle
                    )
                    
                    # Debug output
                    print(f"DEBUG: === OBJECT POSITION CALCULATION ===")
                    print(f"DEBUG: {s3_image_url}")
                    print(f"Image dimensions: {image_width} x {image_height}")
                    print(f"Image center (drone position): {lat:.8f}, {lon:.8f}")
                    print(f"Bbox center: ({bbox_center_x}, {bbox_center_y})")
                    print(f"Offset from center: ({offset_x_px:.1f}px, {offset_y_px:.1f}px)")
                    print(f"Ground sample distance: {gsd:.6f} meters/pixel")
                    print(f"Object position: {object_lat:.8f}, {object_lon:.8f}")
                    print("DEBUG: AprilTag detections found:", len(detections))
                    print("DEBUG: AprilTag result structure:")
                    print(f"DEBUG: {json.dumps(detections, indent=2, default=str)}")    
                    print(f"DEBUG: AprilTag center: ({bbox_center_x}, {bbox_center_y})")
                    print(f"DEBUG: Image dimensions: {image_width} x {image_height}")
                    print(f"DEBUG: AprilTag ID: {detection.get('id', 'unknown')}")
                    print(f"DEBUG: Sensor width: {sensor_width_mm}mm")
                    print(f"DEBUG: Altitude: {alt:.2f}m (GPS units)")
                    print(f"DEBUG: Image width: {image_width}px")
                    print(f"DEBUG: Effective focal length: {effective_focal_length:.2f}mm")
                    print(f"DEBUG: GSD calculation: ({alt:.2f} * {sensor_width_mm}) / ({effective_focal_length:.2f} * {image_width}) = {gsd:.6f}")
                    print(f'DEBUG: Ground sample distance: {gsd:.6f} meters/pixel')
                    print(f"DEBUG: Digital zoom ratio: {digital_zoom_ratio:.2f}x")
                    print(f"DEBUG: 35mm focal length: {focal_length_35mm}mm")
                    print(f"DEBUG: **************************************************" )
                    
                    # Add result for this detection
                    result = {
                        'latitude': object_lat,
                        'longitude': object_lon,
                        'object_lat': object_lat,
                        'object_lon': object_lon,
                        'image_lat': lat,
                        'image_lon': lon,
                        'altitude': alt,
                        'detected_class': 'apriltag',
                        'confidence': 1.0,  # AprilTag detection is binary
                        'apriltag_id': detection.get('id', 'unknown')
                    }
                    results.append(result)
                    logger.info(f"✅ Calculated coordinates for AprilTag ID {apriltag_id}: ({object_lat:.6f}, {object_lon:.6f})")
                
                # Clean up temporary file
                os.remove(local_image_path)
                
                return results
            else:
                logger.error("No AprilTag detections found")
                return []
                
        except subprocess.CalledProcessError as e:
            logger.error(f"AprilTag detection failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AprilTag JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"AprilTag detection error: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting realworld coordinates: {str(e)}")
        return None
    finally:
        # Clean up temporary file if it exists
        if 'local_image_path' in locals() and os.path.exists(local_image_path):
            os.remove(local_image_path)

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

        realworld_coordinates_list = get_realworld_coordinates(s3_image_url)
        logger.info(f"Processing image {image_result.image_name}: Found {len(realworld_coordinates_list) if realworld_coordinates_list else 0} AprilTag detections")
        
        # Handle multiple AprilTag detections - create separate records for each
        if realworld_coordinates_list:
            # Create a separate record for each AprilTag detection
            records_to_insert = []
            for i, realworld_coordinates in enumerate(realworld_coordinates_list):
                # Filter QR results to only include those matching this specific AprilTag ID
                apriltag_id = realworld_coordinates.get('apriltag_id')
                logger.info(f"Processing detection {i+1}/{len(realworld_coordinates_list)}: AprilTag ID {apriltag_id} at ({realworld_coordinates.get('latitude', 'N/A'):.6f}, {realworld_coordinates.get('longitude', 'N/A'):.6f})")
                
                filtered_qr_results = [
                    qr for qr in qr_results_json 
                    if str(qr.get('content', '')) == str(apriltag_id)
                ]
                logger.info(f"Found {len(filtered_qr_results)} matching QR results for AprilTag ID {apriltag_id}")
                
                record = {
                    'task_id': task_id,
                    'filename': image_result.image_name,
                    's3_input_url': s3_image_url,
                    'image_signed_url': image_result.image_url,
                    'timestamp': datetime.now().isoformat(),
                    'latitude': realworld_coordinates.get('latitude'),
                    'longitude': realworld_coordinates.get('longitude'),
                    'qr_results': filtered_qr_results,  # Only QR results for this specific AprilTag
                    'processing_status': 'success' if image_result.success else 'failed',
                    'error_message': image_result.error,
                    'processed_at': datetime.now().isoformat(),
                    'bbox_visualization_url': image_result.bbox_visualization_url
                }
                records_to_insert.append(record)
            
            # Insert all records at once
            if records_to_insert:
                logger.info(f"Attempting to insert {len(records_to_insert)} records into database for {image_result.image_name}")
                try:
                    result = supabase.table('qr_processed_images').insert(records_to_insert).execute()
                    logger.info(f"✅ Successfully inserted {len(records_to_insert)} AprilTag detection records for {image_result.image_name}")
                    logger.info(f"Database response: {result}")
                except Exception as db_error:
                    logger.error(f"❌ Database insertion failed for {image_result.image_name}: {str(db_error)}")
                    logger.error(f"Records that failed to insert: {records_to_insert}")
                    raise
        else:
            # No detections found, create a single record with no coordinates
            logger.info(f"No AprilTag detections found for {image_result.image_name}, creating single record")
            supabase.table('qr_processed_images').insert({
                'task_id': task_id,
                'filename': image_result.image_name,
                's3_input_url': s3_image_url,
                'image_signed_url': image_result.image_url,
                'timestamp': datetime.now().isoformat(),
                'latitude': None,
                'longitude': None,
                'qr_results': qr_results_json,
                'processing_status': 'success' if image_result.success else 'failed',
                'error_message': image_result.error,
                'processed_at': datetime.now().isoformat(),
                'bbox_visualization_url': image_result.bbox_visualization_url
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
    
    # Generate fresh signed URL for bounding box visualization if available
    bbox_visualization_url = None
    if cached_result.get('bbox_visualization_url'):
        try:
            bbox_visualization_url = get_s3_signed_url(cached_result['bbox_visualization_url'])
        except Exception as e:
            logger.warning(f"Failed to generate fresh signed URL for bbox visualization: {str(e)}")
    
    return ImageQRResult(
        image_name=cached_result['filename'],
        image_url=fresh_signed_url,
        image_coordinates=image_coordinates,
        qr_results=qr_results,
        success=cached_result.get('processing_status') == 'success',
        error=cached_result.get('error_message'),
        bbox_visualization_url=bbox_visualization_url
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
        bbox_visualization_url = None
        
        if isinstance(result, list) and len(result) > 0:
            workflow_result = result[0]
            
            crop_outputs = workflow_result.get('crop_output', [])
            predictions = workflow_result.get('model_predictions', {}).get('predictions', [])
            bbox_rf_image = workflow_result.get('bounding_box_visualization', [])
            
            # Debug: Log all available fields in workflow_result
            logger.info(f"Available workflow result fields: {list(workflow_result.keys())}")
            logger.info(f"Found {len(crop_outputs)} QR code detections in {image_name}")
            logger.info(f"Bounding box visualization available: {len(bbox_rf_image) if bbox_rf_image else 0} items")
            
            logger.info(f"OCT27DEBUG: IMAGE URL {s3_image_url}")
            logger.info(f"OCT27DEBUG: {predictions}")
            logger.info(f"OCT27DEBUG: {len(crop_outputs)}")

            # Process bounding box visualization if available
            if bbox_rf_image and len(bbox_rf_image) > 0:
                try:
                    # Generate a task_id for this single image processing
                    task_id = str(uuid.uuid4())
                    # Pass the raw data directly (it's already a list of bytes)
                    bbox_s3_url = upload_bbox_visualization_to_s3(bbox_rf_image, image_name, task_id)
                    bbox_visualization_url = get_s3_signed_url(bbox_s3_url)
                    logger.info(f"✅ Generated bounding box visualization URL: {bbox_visualization_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload bounding box visualization: {str(e)}")
            
            # Process ALL detected QR codes, not just the highest confidence one
            for i, (crop_base64, pred) in enumerate(zip(crop_outputs, predictions)):
                confidence = pred.get('confidence', 0.0)
                logger.info(f"Processing crop {i + 1} with confidence: {confidence:.3f}")
                
                qr_contents = decode_qr_from_base64(crop_base64)
                logger.info(f"OCT27DEBUG: QR contents from crop {i + 1}: {qr_contents}")
                
                if qr_contents:
                    for qr_content in qr_contents:
                        qr_results.append(QRResult(content=qr_content, confidence=confidence))
                        logger.info(f"✅ Decoded QR in {image_name} (crop {i + 1}): {qr_content[:50]}...")
                        logger.info(f"OCT27DEBUG: QR content: {qr_content}")
                else:
                    logger.warning(f"❌ Failed to decode QR from crop {i + 1} in {image_name}")
        
        # Generate signed URL for the image
        image_signed_url = get_s3_signed_url(s3_image_url)
        
        return ImageQRResult(
            image_name=image_name,
            image_url=image_signed_url,
            image_coordinates=image_coordinates,
            qr_results=qr_results,
            success=True,
            bbox_visualization_url=bbox_visualization_url
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

# QR Generation Endpoints
@app.post("/generate-qr", response_model=QRGenerateResponse)
async def generate_qr(request: QRGenerateRequest):
    """Generate QR code with unique ID (VIN assignment happens later)"""
    try:
        logger.info("Generating QR code with unique ID")
        
        # Generate unique QR code ID
        qr_code_id = generate_unique_qr_code_id()
        
        # Generate QR code with the ID (not VIN)
        qr_bytes = generate_qr_code(qr_code_id, request.size)
        
        # Create S3 key
        s3_key = create_s3_key_for_qr(qr_code_id)
        
        # Upload to S3
        s3_url = upload_qr_to_s3(qr_bytes, s3_key)
        
        # Generate signed URL
        image_url = get_s3_signed_url(s3_url)
        
        # Store in database (without VIN initially)
        record_id = store_qr_record(qr_code_id, request.description, s3_url, request.size)
        
        # Create response
        qr_record = QRCodeRecord(
            id=record_id,
            qr_code_id=qr_code_id,
            vin=None,  # Not assigned yet
            description=request.description,
            ai_description=None,  # Will be generated when VIN is assigned
            s3_url=s3_url,
            image_url=image_url,
            created_at=datetime.now(),
            assigned_at=None,
            size=request.size,
            is_active=True
        )
        
        return QRGenerateResponse(
            success=True,
            qr_code=qr_record,
            message=f"QR code generated successfully with ID: {qr_code_id}. Use /reassign-qr to assign a VIN."
        )
        
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate QR code: {str(e)}")

@app.get("/qr-codes", response_model=QRListResponse)
async def list_qr_codes(
    limit: int = 100,
    offset: int = 0
):
    """List all QR codes with signed URLs"""
    try:
        logger.info(f"Fetching QR codes: limit={limit}, offset={offset}")
        
        # Get records from database
        records = get_qr_records(limit, offset)
        
        # Convert to response format with fresh signed URLs
        qr_codes = []
        for record in records:
            # Generate fresh signed URL
            try:
                image_url = get_s3_signed_url(record["s3_url"])
            except Exception as e:
                logger.warning(f"Failed to generate signed URL for {record['id']}: {e}")
                image_url = "URL_EXPIRED"
            
            # Parse assigned_at if it exists
            assigned_at = None
            if record.get("assigned_at"):
                assigned_at = datetime.fromisoformat(record["assigned_at"].replace('Z', '+00:00'))
            
            qr_code = QRCodeRecord(
                id=record["id"],
                qr_code_id=record["qr_code_id"],
                vin=record.get("vin"),
                description=record.get("description"),
                ai_description=record.get("ai_description"),
                s3_url=record["s3_url"],
                image_url=image_url,
                created_at=datetime.fromisoformat(record["created_at"].replace('Z', '+00:00')),
                assigned_at=assigned_at,
                size=record["size"],
                is_active=record.get("is_active", True)
            )
            qr_codes.append(qr_code)
        
        return QRListResponse(
            qr_codes=qr_codes,
            total=len(qr_codes)
        )
        
    except Exception as e:
        logger.error(f"Error listing QR codes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list QR codes: {str(e)}")

@app.get("/qr-codes/{qr_id}")
async def get_qr_code(qr_id: str):
    """Get specific QR code by ID"""
    try:
        record = get_qr_record_by_id(qr_id)
        if not record:
            raise HTTPException(status_code=404, detail="QR code not found")
        
        # Generate fresh signed URL
        image_url = get_s3_signed_url(record["s3_url"])
        
        return QRCodeRecord(
            id=record["id"],
            vin=record["vin"],
            description=record.get("description"),
            ai_description=record.get("ai_description"),
            s3_url=record["s3_url"],
            image_url=image_url,
            created_at=datetime.fromisoformat(record["created_at"].replace('Z', '+00:00')),
            size=record["size"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching QR code {qr_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch QR code: {str(e)}")

@app.delete("/qr-codes/{qr_id}")
async def delete_qr_code(qr_id: str):
    """Delete QR code and its S3 object"""
    try:
        # Get record from database
        record = get_qr_record_by_id(qr_id)
        if not record:
            raise HTTPException(status_code=404, detail="QR code not found")
        
        # Delete from S3
        s3_key = record["s3_url"].replace(f"s3://{os.getenv('S3_BUCKET', 'rcsstoragebucket')}/", "")
        try:
            s3_client = boto3.client('s3')
            s3_client.delete_object(Bucket=os.getenv("S3_BUCKET", "rcsstoragebucket"), Key=s3_key)
            logger.info(f"Deleted S3 object: {s3_key}")
        except Exception as e:
            logger.warning(f"Failed to delete S3 object {s3_key}: {e}")
        
        # Delete from database
        if supabase:
            try:
                supabase.table("qr_codes").delete().eq("id", qr_id).execute()
                logger.info(f"Deleted QR record from database: {qr_id}")
            except Exception as e:
                logger.warning(f"Failed to delete database record {qr_id}: {e}")
        
        return {"success": True, "message": f"QR code {qr_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting QR code {qr_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete QR code: {str(e)}")

@app.post("/reassign-qr", response_model=QRReassignResponse)
async def reassign_qr_code(request: QRReassignRequest):
    """Assign or reassign a VIN to a QR code"""
    try:
        logger.info(f"Reassigning QR code {request.qr_code_id} to VIN {request.vin}")
        
        # Get the QR code record
        record = get_qr_record_by_qr_code_id(request.qr_code_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"QR code {request.qr_code_id} not found")
        
        # Assign the VIN
        updated_record = assign_vin_to_qr_code(request.qr_code_id, request.vin)
        if not updated_record:
            raise HTTPException(status_code=500, detail="Failed to assign VIN to QR code")
        
        # Generate fresh signed URL
        try:
            image_url = get_s3_signed_url(updated_record["s3_url"])
        except Exception as e:
            logger.warning(f"Failed to generate signed URL for {updated_record['id']}: {e}")
            image_url = "URL_EXPIRED"
        
        # Parse assigned_at if it exists
        assigned_at = None
        if updated_record.get("assigned_at"):
            assigned_at = datetime.fromisoformat(updated_record["assigned_at"].replace('Z', '+00:00'))
        
        # Create response
        qr_record = QRCodeRecord(
            id=updated_record["id"],
            qr_code_id=updated_record["qr_code_id"],
            vin=updated_record["vin"],
            description=updated_record.get("description"),
            ai_description=updated_record.get("ai_description"),
            s3_url=updated_record["s3_url"],
            image_url=image_url,
            created_at=datetime.fromisoformat(updated_record["created_at"].replace('Z', '+00:00')),
            assigned_at=assigned_at,
            size=updated_record["size"],
            is_active=updated_record.get("is_active", True)
        )
        
        return QRReassignResponse(
            success=True,
            qr_code=qr_record,
            message=f"QR code {request.qr_code_id} successfully assigned to VIN {request.vin}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reassigning QR code: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reassign QR code: {str(e)}")

@app.post("/generate-and-assign-qr", response_model=QRGenerateAndAssignResponse)
async def generate_and_assign_qr(request: QRGenerateAndAssignRequest):
    """Generate QR code and assign VIN in one step"""
    try:
        logger.info(f"Generating and assigning QR code for VIN: {request.vin}")
        
        # Generate unique QR code ID
        qr_code_id = generate_unique_qr_code_id()
        
        # Generate QR code with the ID
        qr_bytes = generate_qr_code(qr_code_id, request.size)
        
        # Create S3 key
        s3_key = create_s3_key_for_qr(qr_code_id)
        
        # Upload to S3
        s3_url = upload_qr_to_s3(qr_bytes, s3_key)
        
        # Generate signed URL
        image_url = get_s3_signed_url(s3_url)
        
        # Generate AI description for the VIN
        ai_description = generate_vin_description(request.vin)
        
        # Store in database with VIN already assigned
        record_id = str(uuid.uuid4())
        if supabase:
            try:
                record = {
                    "id": record_id,
                    "qr_code_id": qr_code_id,
                    "vin": request.vin,
                    "description": request.description,
                    "ai_description": ai_description,
                    "s3_url": s3_url,
                    "size": request.size,
                    "created_at": datetime.now().isoformat(),
                    "assigned_at": datetime.now().isoformat(),
                    "is_active": True
                }
                
                result = supabase.table("qr_codes").insert(record).execute()
                logger.info(f"Stored QR record with VIN assignment: {record_id}")
            except Exception as e:
                logger.error(f"Error storing QR record: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to store QR record: {str(e)}")
        
        # Create response
        qr_record = QRCodeRecord(
            id=record_id,
            qr_code_id=qr_code_id,
            vin=request.vin,
            description=request.description,
            ai_description=ai_description,
            s3_url=s3_url,
            image_url=image_url,
            created_at=datetime.now(),
            assigned_at=datetime.now(),
            size=request.size,
            is_active=True
        )
        
        return QRGenerateAndAssignResponse(
            success=True,
            qr_code=qr_record,
            message=f"QR code generated and assigned to VIN {request.vin} successfully"
        )
        
    except Exception as e:
        logger.error(f"Error generating and assigning QR code: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate and assign QR code: {str(e)}")

# VIN Tracking Endpoints
@app.get("/folders")
async def list_folders():
    """Get list of all available folders"""
    try:
        if not supabase:
            return {"folders": [], "total": 0}
        
        folder_result = supabase.table("qr_processed_folders").select(
            "folder_name, s3_input_folder_url, total_images, created_at, status"
        ).eq("status", "completed").execute()
        
        folders = []
        for folder in folder_result.data:
            folders.append({
                "folder_name": folder["folder_name"],
                "s3_url": folder["s3_input_folder_url"],
                "total_images": folder["total_images"],
                "created_at": folder["created_at"],
                "status": folder["status"]
            })
        
        # Sort folders by datetime extracted from folder name (most recent first)
        def extract_datetime_from_folder_name(folder_item):
            """Extract datetime from folder name like 'Run September 22 6:07 PM'"""
            folder_name = folder_item["folder_name"]
            try:
                # Parse folder name format: "Run September 22 6:07 PM"
                import re
                from datetime import datetime
                
                # Extract date and time parts
                match = re.match(r"Run (\w+) (\d+) (\d+):(\d+) (AM|PM)", folder_name)
                if match:
                    month_name, day, hour, minute, ampm = match.groups()
                    
                    # Convert month name to number
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    month = month_map.get(month_name, 1)
                    
                    # Convert 12-hour to 24-hour format
                    hour = int(hour)
                    if ampm == 'PM' and hour != 12:
                        hour += 12
                    elif ampm == 'AM' and hour == 12:
                        hour = 0
                    
                    # Create datetime object (assuming current year)
                    current_year = datetime.now().year
                    return datetime(current_year, month, int(day), hour, int(minute))
                
                # Fallback to created_at if parsing fails
                return datetime.fromisoformat(folder_item["created_at"].replace('Z', '+00:00'))
            except:
                # Fallback to created_at if parsing fails
                return datetime.fromisoformat(folder_item["created_at"].replace('Z', '+00:00'))
        
        # Sort by extracted datetime (most recent first)
        folders.sort(key=extract_datetime_from_folder_name, reverse=True)
        
        return {"folders": folders, "total": len(folders)}
        
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list folders: {str(e)}")

@app.get("/vins", response_model=VINListResponse)
async def list_all_vins(folder_filter: Optional[str] = None):
    """Get all unique VINs with summary information
    
    Args:
        folder_filter: Optional comma-separated list of folder names to filter by
    """
    try:
        # Parse folder filter
        folder_list = None
        if folder_filter:
            folder_list = [f.strip() for f in folder_filter.split(",") if f.strip()]
        
        vin_data = get_all_vins_from_processed_images(folder_filter=folder_list)
        
        # Group by VIN
        vin_groups = defaultdict(list)
        for vin_record in vin_data:
            vin_groups[vin_record["vin"]].append(vin_record)
        
        # Create VIN summaries
        vins = []
        for vin, occurrences in vin_groups.items():
            # Sort by timestamp
            occurrences.sort(key=lambda x: x["spotted_at"])
            
            # Get unique folders and sort by datetime
            folders = list(set([v["folder_name"] for v in occurrences]))
            
            # Sort folders by datetime extracted from folder name (most recent first)
            def extract_datetime_from_folder_name(folder_name):
                """Extract datetime from folder name like 'Run September 22 6:07 PM'"""
                try:
                    import re
                    from datetime import datetime
                    
                    # Parse folder name format: "Run September 22 6:07 PM"
                    match = re.match(r"Run (\w+) (\d+) (\d+):(\d+) (AM|PM)", folder_name)
                    if match:
                        month_name, day, hour, minute, ampm = match.groups()
                        
                        # Convert month name to number
                        month_map = {
                            'January': 1, 'February': 2, 'March': 3, 'April': 4,
                            'May': 5, 'June': 6, 'July': 7, 'August': 8,
                            'September': 9, 'October': 10, 'November': 11, 'December': 12
                        }
                        month = month_map.get(month_name, 1)
                        
                        # Convert 12-hour to 24-hour format
                        hour = int(hour)
                        if ampm == 'PM' and hour != 12:
                            hour += 12
                        elif ampm == 'AM' and hour == 12:
                            hour = 0
                        
                        # Create datetime object (assuming current year)
                        current_year = datetime.now().year
                        return datetime(current_year, month, int(day), hour, int(minute))
                    
                    # Fallback to string sorting if parsing fails
                    return folder_name
                except:
                    # Fallback to string sorting if parsing fails
                    return folder_name
            
            # Sort folders by datetime extracted from folder name (most recent first)
            try:
                folders.sort(key=extract_datetime_from_folder_name, reverse=True)
            except Exception as e:
                # Fallback to string sorting if datetime sorting fails
                logger.warning(f"Failed to sort folders by datetime, using string sort: {e}")
                folders.sort(reverse=True)
            
            # Get latest location
            latest_location = None
            latest_occurrence = occurrences[-1]
            if latest_occurrence.get("latitude") and latest_occurrence.get("longitude"):
                latest_location = {
                    "latitude": latest_occurrence["latitude"],
                    "longitude": latest_occurrence["longitude"]
                }
            
            # Get AI description
            ai_description = None
            try:
                if supabase:
                    qr_result = supabase.table("qr_codes").select("ai_description").eq("vin", vin).execute()
                    if qr_result.data:
                        ai_description = qr_result.data[0].get("ai_description")
            except Exception as e:
                logger.warning(f"Could not fetch AI description for VIN {vin}: {e}")
            
            vins.append(VINSummary(
                vin=vin,
                total_spottings=len(occurrences),
                first_spotted=datetime.fromisoformat(occurrences[0]["spotted_at"].replace('Z', '+00:00')),
                last_spotted=datetime.fromisoformat(occurrences[-1]["spotted_at"].replace('Z', '+00:00')),
                folders=folders,
                latest_location=latest_location,
                ai_description=ai_description
            ))
        
        # Sort by last spotted (most recent first)
        vins.sort(key=lambda x: x.last_spotted, reverse=True)
        
        return VINListResponse(
            vins=vins,
            total_vins=len(vins)
        )
        
    except Exception as e:
        logger.error(f"Error listing VINs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list VINs: {str(e)}")

@app.get("/vins/{vin}/history", response_model=VINHistoryResponse)
async def get_vin_history(vin: str, folder_filter: Optional[str] = None):
    """Get detailed history for a specific VIN
    
    Args:
        vin: The VIN to get history for
        folder_filter: Optional comma-separated list of folder names to filter by
    """
    try:
        # Parse folder filter
        folder_list = None
        if folder_filter:
            folder_list = [f.strip() for f in folder_filter.split(",") if f.strip()]
        
        vin_data = get_all_vins_from_processed_images(folder_filter=folder_list)
        vin_occurrences = [v for v in vin_data if v["vin"] == vin]
        
        if not vin_occurrences:
            raise HTTPException(status_code=404, detail=f"VIN {vin} not found")
        
        # Sort by timestamp
        vin_occurrences.sort(key=lambda x: x["spotted_at"])
        
        # Convert to VINHistory objects
        history = []
        for occurrence in vin_occurrences:
            history.append(VINHistory(
                vin=occurrence["vin"],
                folder_name=occurrence["folder_name"],
                folder_s3_url=occurrence["folder_s3_url"],
                image_filename=occurrence["image_filename"],
                image_url=occurrence["image_url"],
                spotted_at=datetime.fromisoformat(occurrence["spotted_at"].replace('Z', '+00:00')),
                latitude=occurrence.get("latitude"),
                longitude=occurrence.get("longitude"),
                confidence=occurrence["confidence"],
                bbox_visualization_url=occurrence.get("bbox_visualization_url")
            ))
        
        return VINHistoryResponse(
            vin=vin,
            history=history,
            total_spottings=len(history)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching VIN history for {vin}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch VIN history: {str(e)}")

@app.get("/vins/{vin}", response_model=VINSummary)
async def get_vin_summary_endpoint(vin: str):
    """Get summary information for a specific VIN"""
    try:
        return get_vin_summary(vin)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching VIN summary for {vin}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch VIN summary: {str(e)}")

@app.get("/map/latest", response_model=MapData)
async def get_latest_folder_map_endpoint():
    """Get map data for the latest processed folder"""
    try:
        map_data = get_latest_folder_map()
        if not map_data:
            raise HTTPException(status_code=404, detail="No completed folders found")
        return map_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest folder map: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest folder map: {str(e)}")

@app.get("/dashboard", response_model=VINDashboardResponse)
async def get_vin_dashboard(folders: Optional[str] = None):
    """Get comprehensive VIN dashboard with all VINs and latest map
    
    Args:
        folders: Optional comma-separated list of folder names to filter by.
                If not provided, uses all folders.
    """
    try:
        # Parse folders parameter
        folder_filter = None
        if folders:
            folder_filter = [f.strip() for f in folders.split(",") if f.strip()]
            logger.info(f"Filtering dashboard by folders: {folder_filter}")
        
        # Get all VINs (with optional folder filtering)
        vin_response = await list_all_vins(folder_filter=",".join(folder_filter) if folder_filter else None)
        
        # Get latest map (with optional folder filtering)
        latest_map = get_latest_folder_map(folder_filter=folder_filter)
        
        # Get total folders count (with optional filtering)
        total_folders = 0
        if supabase:
            try:
                if folder_filter:
                    folder_result = supabase.table("qr_processed_folders").select("id", count="exact").in_("folder_name", folder_filter).execute()
                else:
                    folder_result = supabase.table("qr_processed_folders").select("id", count="exact").execute()
                total_folders = folder_result.count or 0
            except Exception as e:
                logger.warning(f"Could not fetch total folders count: {e}")
        
        return VINDashboardResponse(
            vins=vin_response.vins,
            latest_map=latest_map,
            total_vins=vin_response.total_vins,
            total_folders=total_folders
        )
        
    except Exception as e:
        logger.error(f"Error fetching VIN dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch VIN dashboard: {str(e)}")

@app.get("/vins/{vin}/movement", response_model=VINMovementPath)
async def get_vin_movement_path_endpoint(vin: str, folder_filter: Optional[str] = None):
    """Get movement path for a specific VIN showing GPS trail over time
    
    Args:
        vin: The VIN to get movement path for
        folder_filter: Optional comma-separated list of folder names to filter by
    """
    try:
        # Parse folder filter
        folder_list = None
        if folder_filter:
            folder_list = [f.strip() for f in folder_filter.split(",") if f.strip()]
        
        return get_vin_movement_path(vin, folder_filter=folder_list)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching movement path for {vin}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch movement path: {str(e)}")

@app.post("/detect-apriltag", response_model=AprilTagDetectionResponse)
async def detect_apriltag(request: AprilTagDetectionRequest):
    """Detect AprilTags in an image and return results in JSON or binary format
    
    Args:
        request: AprilTagDetectionRequest containing base64 image and output format
    """
    try:
        logger.info(f"Processing AprilTag detection request with format: {request.format}")
        
        # Decode the base64 image
        try:
            # Strip data URL prefix if present
            image_data = request.image
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Log original image properties
            img = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Original image mode: {img.mode}, size: {img.size}")
            
            # Use enhanced detection with preprocessing
            detections = detect_apriltags_enhanced(image_bytes)
            logger.info(f"Enhanced AprilTag detector found {len(detections)} detections")
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        if not detections:
            return AprilTagDetectionResponse(
                success=True,
                detections=[],
                total_detections=0,
                message="No AprilTags detected in the image"
            )
        
        # Convert to our response format
        apriltag_detections = []
        for detection in detections:
            apriltag_detections.append(AprilTagDetection(
                id=detection.get('id', 0),
                center=detection.get('center', {'x': 0.0, 'y': 0.0}),
                corners=detection.get('corners', []),
                confidence=detection.get('confidence', 1.0)
            ))
        
        # Handle binary format if requested
        binary_data = None
        if request.format.lower() == "binary":
            # Convert detections to binary format
            binary_data = json.dumps([{
                'id': det.id,
                'center': det.center,
                'corners': det.corners,
                'confidence': det.confidence
            } for det in apriltag_detections]).encode('utf-8')
        
        return AprilTagDetectionResponse(
            success=True,
            detections=apriltag_detections,
            total_detections=len(apriltag_detections),
            message=f"Successfully detected {len(apriltag_detections)} AprilTag(s)",
            binary_data=binary_data
        )
        
    except subprocess.CalledProcessError as e:
        logger.error(f"AprilTag detection failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"AprilTag detection failed: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AprilTag JSON response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse detection results: {str(e)}")
    except Exception as e:
        logger.error(f"AprilTag detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AprilTag detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
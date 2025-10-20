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
import qrcode
from io import BytesIO
import openai
from collections import defaultdict
import math

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
    real_world_coordinates: Optional[List[Dict[str, float]]] = None
    success: bool
    error: Optional[str] = None

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
    """Extract GPS coordinates and drone metadata from image EXIF data using exifread"""
    try:
        # Get signed URL and download image
        https_url = get_s3_signed_url(s3_url)
        response = requests.get(https_url)
        response.raise_for_status()
        
        # Use exifread for comprehensive EXIF extraction
        import exifread
        image_file = io.BytesIO(response.content)
        tags = exifread.process_file(image_file, details=True)
        
        # Extract GPS coordinates
        latitude = None
        longitude = None
        altitude = None
        
        # Extract GPS data using exifread
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            # Convert DMS to decimal degrees
            lat_dms = tags['GPS GPSLatitude']
            lon_dms = tags['GPS GPSLongitude']
            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N'))
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E'))
            
            # Convert to decimal degrees
            lat_decimal = float(lat_dms.values[0]) + float(lat_dms.values[1])/60.0 + float(lat_dms.values[2])/3600.0
            lon_decimal = float(lon_dms.values[0]) + float(lon_dms.values[1])/60.0 + float(lon_dms.values[2])/3600.0
            
            if lat_ref == 'S':
                lat_decimal = -lat_decimal
            if lon_ref == 'W':
                lon_decimal = -lon_decimal
                
            latitude = lat_decimal
            longitude = lon_decimal
        
        # Extract altitude
        if 'GPS GPSAltitude' in tags:
            altitude_tag = tags['GPS GPSAltitude']
            altitude = float(altitude_tag.values[0]) if hasattr(altitude_tag, 'values') else float(altitude_tag)
        
        if not latitude or not longitude:
            logger.warning("No GPS coordinates found in EXIF data")
            return None
        
        # Extract drone metadata from EXIF
        drone_metadata = {}
        
        # Get altitude
        if altitude:
            drone_metadata['altitude'] = altitude
        
        # Extract camera parameters from EXIF
        if 'EXIF FocalLength' in tags:
            focal_tag = tags['EXIF FocalLength']
            focal_length = float(focal_tag.values[0]) if hasattr(focal_tag, 'values') else float(focal_tag)
            drone_metadata['focal_length'] = focal_length
        
        if 'EXIF DigitalZoomRatio' in tags:
            zoom_tag = tags['EXIF DigitalZoomRatio']
            digital_zoom = float(zoom_tag.values[0]) if hasattr(zoom_tag, 'values') else float(zoom_tag)
            drone_metadata['digital_zoom_ratio'] = digital_zoom
        
        if 'EXIF ExifImageWidth' in tags:
            width_tag = tags['EXIF ExifImageWidth']
            image_width = int(width_tag.values[0]) if hasattr(width_tag, 'values') else int(width_tag)
            drone_metadata['image_width'] = image_width
        
        if 'EXIF ExifImageLength' in tags:
            height_tag = tags['EXIF ExifImageLength']
            image_height = int(height_tag.values[0]) if hasattr(height_tag, 'values') else int(height_tag)
            drone_metadata['image_height'] = image_height
        
        # Calculate FOV from focal length (approximate)
        if 'focal_length' in drone_metadata:
            # DJI M3TD has approximately 12.8° FOV at 29.9mm
            focal_length = drone_metadata['focal_length']
            base_fov = 12.8
            base_focal = 29.9
            drone_metadata['fov'] = base_fov * (base_focal / focal_length)
        else:
            drone_metadata['fov'] = 12.8  # Default FOV
        
        # Set default values for missing drone orientation data
        # (These would need to be extracted from DJI-specific EXIF tags if available)
        drone_metadata['yaw'] = 0.0
        drone_metadata['pitch'] = 0.0
        drone_metadata['roll'] = 0.0
        drone_metadata['gimbal_yaw'] = 0.0
        drone_metadata['gimbal_pitch'] = -90.0  # Nadir (pointing down)
        drone_metadata['gimbal_roll'] = 0.0
        
        # Set default flight dynamics
        drone_metadata['flight_x_speed'] = 0.0
        drone_metadata['flight_y_speed'] = 0.0
        drone_metadata['flight_z_speed'] = 0.0
        drone_metadata['relative_altitude'] = 0.0
        drone_metadata['sensor_temperature'] = 0.0
        drone_metadata['gps_status'] = str(tags.get('GPS GPSStatus', 'Unknown'))
        
        logger.info(f"📍 GPS coordinates extracted: {latitude:.6f}, {longitude:.6f}")
        logger.info(f"📍 Drone altitude: {drone_metadata.get('altitude', 'unknown')}m")
        logger.info(f"📍 Camera FOV: {drone_metadata.get('fov', 'unknown')}°")
        logger.info(f"📍 Digital zoom: {drone_metadata.get('digital_zoom_ratio', 'unknown')}x")
        
        return {
            'latitude': latitude,
            'longitude': longitude,
            'drone_metadata': drone_metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to extract GPS from image {s3_url}: {str(e)}")
        return None

def convert_to_degrees(value):
    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        d, m, s = value
        return float(d) + float(m)/60 + float(s)/3600
    return 0.0

def calculate_real_world_coordinates(
    image_coordinates: Dict[str, float],
    predictions: List[Dict],
    image_width: int = 4000,
    image_height: int = 3000
) -> List[Dict[str, Any]]:
    """
    Calculate real-world GPS coordinates for QR code detections using actual drone metadata.
    
    Args:
        image_coordinates: Drone GPS coordinates and metadata from EXIF
        predictions: Roboflow predictions with bounding boxes
        image_width: Image width in pixels (fallback)
        image_height: Image height in pixels (fallback)
        
    Returns:
        List of QR results with real-world coordinates
    """
    if not image_coordinates or not predictions:
        return []
    
    real_world_results = []
    drone_lat = image_coordinates['latitude']
    drone_lon = image_coordinates['longitude']
    
    # Get drone metadata from EXIF
    drone_metadata = image_coordinates.get('drone_metadata', {})
    
    # Use actual values from EXIF, with fallbacks
    altitude = drone_metadata.get('altitude', 30.0)
    relative_altitude = drone_metadata.get('relative_altitude', 0.0)
    
    # Use gimbal yaw if available (more accurate for camera orientation), otherwise drone yaw
    yaw = drone_metadata.get('gimbal_yaw', drone_metadata.get('yaw', 0.0))
    pitch = drone_metadata.get('gimbal_pitch', drone_metadata.get('pitch', 0.0))
    roll = drone_metadata.get('gimbal_roll', drone_metadata.get('roll', 0.0))
    
    # Calculate effective FOV considering digital zoom
    base_fov = drone_metadata.get('fov', 12.8)
    digital_zoom_ratio = drone_metadata.get('digital_zoom_ratio', 1.0)
    fov = base_fov / digital_zoom_ratio  # Digital zoom reduces effective FOV
    
    actual_width = drone_metadata.get('image_width', image_width)
    actual_height = drone_metadata.get('image_height', image_height)
    
    # Get flight dynamics for potential motion compensation
    flight_x_speed = drone_metadata.get('flight_x_speed', 0.0)
    flight_y_speed = drone_metadata.get('flight_y_speed', 0.0)
    flight_z_speed = drone_metadata.get('flight_z_speed', 0.0)
    
    # Get environmental factors
    sensor_temp = drone_metadata.get('sensor_temperature', 0.0)
    gps_status = drone_metadata.get('gps_status', 'Unknown')
    
    logger.info(f"📍 Using enhanced drone metadata:")
    logger.info(f"   Altitude: {altitude}m (relative: {relative_altitude}m)")
    logger.info(f"   Orientation: yaw={yaw}°, pitch={pitch}°, roll={roll}°")
    logger.info(f"   FOV: {fov:.2f}° (base: {base_fov}°, zoom: {digital_zoom_ratio}x)")
    logger.info(f"   GPS Status: {gps_status}, Sensor Temp: {sensor_temp}°C")
    logger.info(f"   Flight Speed: X={flight_x_speed}m/s, Y={flight_y_speed}m/s, Z={flight_z_speed}m/s")
    
    for prediction in predictions:
        # Extract bounding box coordinates from Roboflow prediction
        # Roboflow format: x, y are center coordinates in pixels
        detection_center_x = prediction.get('x', actual_width / 2)  # Center x in pixels
        detection_center_y = prediction.get('y', actual_height / 2)  # Center y in pixels
        bbox_width = prediction.get('width', 0)
        bbox_height = prediction.get('height', 0)
        
        logger.info(f"📍 QR detection at pixel ({detection_center_x:.1f}, {detection_center_y:.1f}) size {bbox_width}x{bbox_height}")
        
        # Calculate GPS coordinates using pixel to GPS conversion
        try:
            # Step 1: Normalize detection coordinates to [-1, 1] relative to image center
            center_x = actual_width / 2
            center_y = actual_height / 2
            offset_x = detection_center_x - center_x
            offset_y = detection_center_y - center_y
            normalized_x = offset_x / (actual_width / 2)
            normalized_y = offset_y / (actual_height / 2)
            
            # Step 2: Convert normalized offsets to camera angles using actual FOV
            fov_rad = math.radians(fov)
            horizontal_angle = normalized_x * (fov_rad / 2)
            vertical_angle = normalized_y * (fov_rad / 2)
            
            # Step 3: Estimate ground offsets using actual altitude and camera angles
            # Use gimbal pitch/roll for camera orientation, but limit extreme values
            gimbal_pitch = drone_metadata.get('gimbal_pitch', pitch)
            gimbal_roll = drone_metadata.get('gimbal_roll', roll)
            
            # Limit pitch/roll to reasonable values to avoid extreme calculations
            gimbal_pitch = max(-45, min(45, gimbal_pitch))  # Limit to ±45°
            gimbal_roll = max(-45, min(45, gimbal_roll))    # Limit to ±45°
            
            pitch_rad = math.radians(gimbal_pitch)
            roll_rad = math.radians(gimbal_roll)
            
            # Calculate ground offsets with moderate pitch/roll compensation
            dx = altitude * math.tan(horizontal_angle) / max(0.1, math.cos(pitch_rad))
            dy = altitude * math.tan(vertical_angle) / max(0.1, math.cos(roll_rad))
            
            # Step 4: Rotate offsets by actual gimbal yaw (inverted for correct coordinate system)
            yaw_rad = math.radians(yaw + 180)  # Invert yaw for correct coordinate system
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            rotated_dx = dx * cos_yaw - dy * sin_yaw
            rotated_dy = dx * sin_yaw + dy * cos_yaw
            
            # Use the rotated values directly (skip additional pitch/roll compensation for now)
            final_dx = rotated_dx
            final_dy = rotated_dy
            
            # Step 5: Convert meter offsets to GPS deltas using Earth radius and latitude
            EARTH_RADIUS = 6371000  # Earth radius in meters
            lat_rad = math.radians(drone_lat)
            latitude_delta = math.degrees(final_dy / EARTH_RADIUS)
            longitude_delta = math.degrees(final_dx / (EARTH_RADIUS * math.cos(lat_rad)))
            
            # Step 6: Add deltas to drone GPS to get estimated detection coordinates
            qr_lat = drone_lat + latitude_delta
            qr_lon = drone_lon + longitude_delta
            
            real_world_results.append({
                'latitude': qr_lat,
                'longitude': qr_lon,
                'confidence': prediction.get('confidence', 0.0)
            })
            
        except Exception as e:
            logger.warning(f"Failed to calculate real-world coordinates: {e}")
            continue
    
    return real_world_results

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

# QR Generation Helper Functions
def generate_qr_code(data: str, size: int = 10) -> bytes:
    """Generate ultra-high-resolution QR code as PNG bytes for professional A4 printing"""
    # For professional A4 printing at 300 DPI, we want the QR code to be 3-4 inches
    # That's 900-1200 pixels, so with a typical QR code being ~25x25 modules
    # We need box_size of 36-48 for ultra-high resolution A4 printing
    
    # Ultra-high resolution settings for professional printing
    print_box_size = max(60, size * 5)  # Minimum 60 for ultra-high res, or 5x the requested size
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction for maximum reliability
        box_size=print_box_size,
        border=4,  # Minimal border to maximize QR code size
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes with maximum quality settings
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG', optimize=False, compress_level=0, dpi=(300, 300))
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def create_s3_key_for_qr(qr_code_id: str) -> str:
    """Create S3 key for QR code"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = qr_code_id.replace("/", "_").replace("\\", "_")
    return f"qr_codes/{safe_id}_{timestamp}.png"

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
            "id, filename, s3_input_url, image_signed_url, timestamp, latitude, longitude, qr_results, task_id"
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
                            "task_id": image["task_id"]
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
        
        # Sort by folder name to get the most recent drone run chronologically
        # Folder names like "Run September 22 6:07 PM" sort correctly as strings
        folders = sorted(folder_result.data, key=lambda x: x["folder_name"], reverse=True)
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

        # Note: Real-world coordinate calculation moved to process_single_image where predictions are available
        
        supabase.table('qr_processed_images').insert({
            'task_id': task_id,
            'filename': image_result.image_name,
            's3_input_url': s3_image_url,
            'image_signed_url': image_result.image_url,
            'timestamp': datetime.now().isoformat(),
            # Store real-world coordinates if available, otherwise use image coordinates
            'latitude': (image_result.real_world_coordinates[0]['latitude'] if image_result.real_world_coordinates 
                        else image_result.image_coordinates.get('latitude') if image_result.image_coordinates else None),
            'longitude': (image_result.real_world_coordinates[0]['longitude'] if image_result.real_world_coordinates 
                         else image_result.image_coordinates.get('longitude') if image_result.image_coordinates else None),
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
            
            # Calculate real-world coordinates for QR detections
            real_world_coordinates = []
            if image_coordinates and predictions:
                real_world_coordinates = calculate_real_world_coordinates(
                    image_coordinates,
                    predictions,
                    4000,  # image_width
                    3000   # image_height
                )
                logger.info(f"📍 Calculated {len(real_world_coordinates)} real-world coordinates")
            
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
            real_world_coordinates=real_world_coordinates,
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
                confidence=occurrence["confidence"]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
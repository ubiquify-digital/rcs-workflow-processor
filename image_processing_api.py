# Image Processing API for S3 Folders
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from supabase import create_client, Client
from inference_sdk import InferenceHTTPClient
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "MQ1Wd6PJGMPBMCvxsCS6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "rcs-k9i1w")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "awais-detect-trash")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Image Processing API", description="API for processing image folders with object detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for frontend access
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response models
class ImageFolderProcessRequest(BaseModel):
    s3_folder_url: str  # s3://bucket/folder/path/
    
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
    s3_output_folder_url: Optional[str] = None
    error: Optional[str] = None

class ImageCompareRequest(BaseModel):
    folder1_task_id: str
    folder2_task_id: str
    distance_threshold_meters: float = 1  # Default 1 meters

class MultifolderAnalysisRequest(BaseModel):
    task_ids: List[str]  # List of task IDs to analyze
    distance_threshold_meters: float = 1  # Clustering radius in meters
    min_time_gap_minutes: float = 5  # Minimum time between frames to consider for violation
    
class ImageMatch(BaseModel):
    image1: Dict[str, Any]
    image2: Dict[str, Any]
    distance_meters: float

class ImageCompareResponse(BaseModel):
    matches: List[ImageMatch]
    total_matches: int
    folder1_images: int
    folder2_images: int
    distance_threshold_meters: float

class DetectionChange(BaseModel):
    object_type: str
    change_type: str  # 'car_left_trash', 'car_dumped_trash', 'car_arrived_dumped', 'additional_dumping'
    before_count: int
    after_count: int
    confidence_avg: Optional[float] = None

class EnrichedImageMatch(BaseModel):
    image1: Dict[str, Any]
    image2: Dict[str, Any] 
    distance_meters: float
    time_difference_minutes: float
    detection_changes: List[DetectionChange]
    change_summary: str

class EnrichedCompareResponse(BaseModel):
    matches: List[EnrichedImageMatch]
    total_matches: int
    folder1_images: int
    folder2_images: int
    distance_threshold_meters: float
    change_statistics: Dict[str, int]

class SrtGenerateRequest(BaseModel):
    task_id: str
    font_size: int = 28
    coordinate_precision: int = 6  # Decimal places for coordinates


class SrtGenerateResponse(BaseModel):
    srt_content: str
    total_entries: int
    duration_seconds: int
    gps_range: Dict[str, Dict[str, float]]  # lat/lon min/max
    images_used: int
    interpolated_points: int


class DumpingViolation(BaseModel):
    cluster_id: int
    violation_type: str  # 'car_left_trash', 'car_dumped_trash', etc.
    location: Dict[str, float]  # lat, lon, center of cluster
    before_frame: Dict[str, Any]  # Image with car (and maybe trash) - includes folder_name, signed_input_url, signed_output_url
    after_frame: Dict[str, Any]   # Image with violation (trash left behind, etc.) - includes folder_name, signed_input_url, signed_output_url
    before_folder: str  # Folder name of the before frame
    after_folder: str   # Folder name of the after frame
    vehicle_plate: Optional[str]  # License plate number if detected
    time_difference_minutes: float
    distance_meters: float  # Distance between before and after GPS coordinates
    trash_count: int
    description: str

class MultifolderAnalysisResponse(BaseModel):
    violations: List[DumpingViolation]
    total_violations: int
    clusters_analyzed: int
    total_frames: int
    folders_analyzed: List[str]
    analysis_summary: Dict[str, int]  # violation type counts

# In-memory task storage
tasks = {}

def add_signed_urls_to_images(images: list) -> list:
    """Add signed URLs to a list of image objects"""
    try:
        s3_client = boto3.client('s3')
        for image in images:
            # Generate signed URL for input
            if image.get('s3_input_url'):
                s3_input_url = image['s3_input_url']
                if s3_input_url.startswith('s3://'):
                    # Parse S3 URL
                    s3_parts = s3_input_url.replace('s3://', '').split('/', 1)
                    if len(s3_parts) == 2:
                        bucket, key = s3_parts
                        try:
                            signed_input_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={'Bucket': bucket, 'Key': key},
                                ExpiresIn=3600  # 1 hour
                            )
                            image['signed_input_url'] = signed_input_url
                        except Exception as e:
                            logger.error(f"Error generating signed input URL for {s3_input_url}: {str(e)}")
                            image['signed_input_url'] = None
                    else:
                        image['signed_input_url'] = None
                else:
                    image['signed_input_url'] = None
            else:
                image['signed_input_url'] = None
                
            # Generate signed URL for output
            if image.get('s3_output_url'):
                s3_output_url = image['s3_output_url']
                if s3_output_url.startswith('s3://'):
                    # Parse S3 URL
                    s3_parts = s3_output_url.replace('s3://', '').split('/', 1)
                    if len(s3_parts) == 2:
                        bucket, key = s3_parts
                        try:
                            signed_output_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={'Bucket': bucket, 'Key': key},
                                ExpiresIn=3600  # 1 hour
                            )
                            image['signed_output_url'] = signed_output_url
                        except Exception as e:
                            logger.error(f"Error generating signed output URL for {s3_output_url}: {str(e)}")
                            image['signed_output_url'] = None
                    else:
                        image['signed_output_url'] = None
                else:
                    image['signed_output_url'] = None
            else:
                image['signed_output_url'] = None
                
    except Exception as e:
        logger.error(f"Error adding signed URLs to images: {str(e)}")
        # Add null signed URLs to all images if there's an error
        for image in images:
            image['signed_input_url'] = None
            image['signed_output_url'] = None
    
    return images

def analyze_detection_changes(image1: dict, image2: dict) -> tuple[List[DetectionChange], str]:
    """Detect car dumping violations by analyzing car and trash patterns between two images"""
    changes = []
    
    try:
        # Extract detection data
        detections1 = image1.get('detections', [])
        detections2 = image2.get('detections', [])
        
        if not detections1 or not detections2:
            return [], "No detection data available"
        
        # Get predictions from both images
        def extract_predictions(detections):
            predictions = {}
            for detection in detections:
                if isinstance(detection, dict):
                    # Car predictions
                    car_preds = detection.get('car_model_predictions', {}).get('predictions', [])
                    if car_preds:
                        predictions['car'] = len(car_preds)

                    # Trash predictions
                    trash_preds = detection.get('trash_model_predictions', {}).get('predictions', [])
                    if trash_preds:
                        predictions['trash'] = len(trash_preds)
                    
                    # License plate predictions  
                    plate_preds = detection.get('license_plate_model_predictions', {}).get('predictions', [])
                    if plate_preds:
                        predictions['license_plate'] = len(plate_preds)
                    
                    # OpenAI detections (if available)
                    openai_preds = detection.get('open_ai', [])
                    if openai_preds:
                        for pred in openai_preds:
                            if isinstance(pred, dict):
                                obj_type = pred.get('class', pred.get('label', 'unknown'))
                                predictions[obj_type] = predictions.get(obj_type, 0) + 1
            
            return predictions
        
        preds1 = extract_predictions(detections1)
        preds2 = extract_predictions(detections2)
        
        # Get car and trash counts
        cars1 = preds1.get('car', 0)
        cars2 = preds2.get('car', 0)
        trash1 = preds1.get('trash', 0)
        trash2 = preds2.get('trash', 0)
        
        # Check for car dumping patterns (inspired by the reference code)
        dumping_detected = False
        dumping_description = ""
        
        # Pattern 1: Car+Trash initially → Trash only later (car left trash behind)
        if cars1 > 0 and trash1 > 0 and cars2 == 0 and trash2 > 0:
            dumping_detected = True
            dumping_description = f"🚗🗑️ DUMPING DETECTED: Car left {trash2} trash item(s) behind"
            changes.append(DetectionChange(
                object_type="dumping_violation",
                change_type="car_left_trash",
                before_count=cars1,
                after_count=trash2
            ))
        
        # Pattern 2: Car initially → Car+Trash later (car brought/dumped trash)
        # elif cars1 > 0 and trash1 == 0 and cars2 > 0 and trash2 > 0:
        #     dumping_detected = True
        #     dumping_description = f"🚗🗑️ DUMPING DETECTED: Car dumped {trash2} trash item(s)"
        #     changes.append(DetectionChange(
        #         object_type="dumping_violation",
        #         change_type="car_dumped_trash",
        #         before_count=0,
        #         after_count=trash2
        #     ))
        
        # NOT NEEDED FOR NOW: Pattern 3: No objects initially → Car+Trash later (new dumping event)
        # elif cars1 == 0 and trash1 == 0 and cars2 > 0 and trash2 > 0:
        #     dumping_detected = True
        #     dumping_description = f"🚗🗑️ DUMPING DETECTED: Car arrived and dumped {trash2} trash item(s)"
        #     changes.append(DetectionChange(
        #         object_type="dumping_violation",
        #         change_type="car_arrived_dumped",
        #         before_count=0,
        #         after_count=trash2
        #     ))
        
        # Pattern 4: Car+Trash initially → More trash later (additional dumping)
        # elif cars1 > 0 and trash1 > 0 and trash2 > trash1:
        #     dumping_detected = True
        #     additional_trash = trash2 - trash1
        #     dumping_description = f"🚗🗑️ DUMPING DETECTED: Additional {additional_trash} trash item(s) dumped"
        #     changes.append(DetectionChange(
        #         object_type="dumping_violation", 
        #         change_type="additional_dumping",
        #         before_count=trash1,
        #         after_count=trash2
        #     ))
        
        # Generate summary - focus on dumping detection as primary use case
        if dumping_detected:
            summary = dumping_description
        else:
            summary = "No dumping violations detected"
        
        return changes, summary
        
    except Exception as e:
        logger.error(f"Error analyzing detection changes: {str(e)}")
        return [], f"Error analyzing changes: {str(e)}"

def interpolate_coordinates(gps_data: List[Tuple[datetime, float, float]], target_time: datetime) -> Tuple[float, float]:
    """Interpolate GPS coordinates for a given time"""
    if not gps_data:
        return 0.0, 0.0
    
    # Find the two closest points
    before_point = None
    after_point = None
    
    for i, (time, lat, lon) in enumerate(gps_data):
        if time <= target_time:
            before_point = (time, lat, lon)
        if time >= target_time and after_point is None:
            after_point = (time, lat, lon)
            break
    
    # If we only have one point or target is outside range
    if before_point is None:
        return after_point[1], after_point[2] if after_point else (0.0, 0.0)
    if after_point is None:
        return before_point[1], before_point[2]
    
    # If points are the same, no interpolation needed
    if before_point[0] == after_point[0]:
        return before_point[1], before_point[2]
    
    # Linear interpolation
    time_diff = (after_point[0] - before_point[0]).total_seconds()
    target_diff = (target_time - before_point[0]).total_seconds()
    ratio = target_diff / time_diff if time_diff > 0 else 0
    
    lat = before_point[1] + (after_point[1] - before_point[1]) * ratio
    lon = before_point[2] + (after_point[2] - before_point[2]) * ratio
    
    return lat, lon

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def generate_srt_content(images: List[dict], font_size: int = 28, precision: int = 6) -> Tuple[str, dict]:
    """Generate SRT subtitle content from image GPS data - one entry per image"""
    from datetime import datetime, timedelta
    
    # Extract GPS data with timestamps
    gps_data = []
    for img in images:
        if img.get('latitude') and img.get('longitude') and img.get('timestamp'):
            try:
                # Parse timestamp
                timestamp_str = img['timestamp']
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str.replace('Z', '+00:00')
                dt = datetime.fromisoformat(timestamp_str)
                
                lat = float(img['latitude'])
                lon = float(img['longitude'])
                filename = img.get('filename', 'unknown')
                gps_data.append((dt, lat, lon, filename))
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing timestamp or coordinates for image {img.get('filename', 'unknown')}: {e}")
                continue
    
    if not gps_data:
        raise ValueError("No valid GPS data found in images")
    
    # Sort by timestamp
    gps_data.sort(key=lambda x: x[0])
    
    # Generate SRT entries - one per image
    srt_entries = []
    num_images = len(gps_data)
    
    # Fixed duration: 1 second per image
    interval = 1.0
    
    for i, (dt, lat, lon, filename) in enumerate(gps_data):
        # Calculate timing for this image
        seconds_offset = i * interval
        
        # Create SRT entry
        entry_num = i + 1
        time_start = format_srt_time(seconds_offset)
        time_end = format_srt_time(seconds_offset + interval)
        
        # Format coordinates with specified precision
        lat_str = f"{lat:.{precision}f}"
        lon_str = f"{lon:.{precision}f}"
        
        srt_entry = f"""{entry_num}
{time_start} --> {time_end}
<font size="{font_size}">GPS: Lat {lat_str}, Lon {lon_str}</font>

"""
        srt_entries.append(srt_entry)
    
    # Calculate GPS range
    all_lats = [data[1] for data in gps_data]
    all_lons = [data[2] for data in gps_data]
    
    gps_range = {
        "latitude": {
            "min": min(all_lats),
            "max": max(all_lats)
        },
        "longitude": {
            "min": min(all_lons),
            "max": max(all_lons)
        }
    }
    
    # Calculate total video duration (1 second per image)
    total_duration = num_images * 1.0
    
    stats = {
        "images_used": len(gps_data),
        "interpolated_points": 0,  # No interpolation needed since we use actual image data
        "gps_range": gps_range,
        "total_duration_seconds": total_duration
    }
    
    return ''.join(srt_entries), stats

class ImageProcessor:
    def __init__(self, task_id: str, s3_input_folder_url: str):
        self.task_id = task_id
        self.temp_dir = None
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.s3_input_folder_url = s3_input_folder_url
        self.s3_output_folder_url = None
        
        # Parse input folder to determine output location
        # s3://bucket/folder/path/ -> s3://bucket/folder/path/outputs/
        self.s3_output_folder_url = s3_input_folder_url.rstrip('/') + '/outputs/'
        
    def generate_signed_url(self, bucket: str, key: str, expiry_hours: int = 1) -> str:
        """Generate a signed URL for S3 object access"""
        try:
            s3_client = boto3.client('s3')
            signed_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiry_hours * 3600
            )
            return signed_url
        except Exception as e:
            logger.error(f"Error generating signed URL for s3://{bucket}/{key}: {str(e)}")
            return None
        
    def download_images_from_s3_folder(self, s3_folder_url: str) -> List[str]:
        """Download all images from S3 folder and return local paths"""
        try:
            # Parse S3 URL
            s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
            bucket_name = s3_parts[0]
            folder_prefix = s3_parts[1] + '/' if len(s3_parts) > 1 else ''
            
            # List objects in folder (only immediate files, not subfolders)
            s3_client = boto3.client('s3')
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix, Delimiter='/')
            
            if 'Contents' not in response:
                raise ValueError("No files found in the specified folder")
            
            # Filter for image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_objects = [
                obj for obj in response['Contents'] 
                if any(obj['Key'].lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_objects:
                raise ValueError("No image files found in the folder")
            
            self.total_count = len(image_objects)
            logger.info(f"Found {self.total_count} images in folder")
            
            # Create temp directory for images
            images_dir = os.path.join(self.temp_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            local_paths = []
            for i, obj in enumerate(image_objects):
                # Download each image
                local_filename = os.path.basename(obj['Key'])
                local_path = os.path.join(images_dir, local_filename)
                
                s3_client.download_file(bucket_name, obj['Key'], local_path)
                local_paths.append(local_path)
                
                # Update progress
                tasks[self.task_id]["progress"] = f"Downloaded {i+1}/{self.total_count} images"
            
            # Sort images by filename (assuming timestamp-based naming)
            local_paths.sort()
            return local_paths
            
        except Exception as e:
            logger.error(f"Error downloading images: {str(e)}")
            raise
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract timestamp and GPS coordinates from image EXIF data"""
        try:
            with Image.open(image_path) as img:
                exif_dict = img.getexif()
                
            metadata = {
                'filename': os.path.basename(image_path),
                'timestamp': None,
                'latitude': None,
                'longitude': None,
                'altitude': None
            }
            
            if not exif_dict:
                logger.warning(f"No EXIF data found in {os.path.basename(image_path)}")
                return metadata
            
            # Extract timestamp
            for tag_id, value in exif_dict.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                    try:
                        metadata['timestamp'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
                        break  # Use first valid timestamp found
                    except ValueError:
                        continue
            
            # Extract GPS data
            gps_ifd = exif_dict.get_ifd(0x8825)  # GPS IFD tag
            if gps_ifd:
                logger.info(f"Found GPS data in {os.path.basename(image_path)}")
                
                def convert_to_degrees(value):
                    """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
                    if isinstance(value, (tuple, list)) and len(value) == 3:
                        d, m, s = value
                        return float(d) + float(m)/60 + float(s)/3600
                    return 0.0
                
                # Get GPS coordinates
                gps_latitude = gps_ifd.get(2)  # GPSLatitude
                gps_latitude_ref = gps_ifd.get(1)  # GPSLatitudeRef
                gps_longitude = gps_ifd.get(4)  # GPSLongitude  
                gps_longitude_ref = gps_ifd.get(3)  # GPSLongitudeRef
                gps_altitude = gps_ifd.get(6)  # GPSAltitude
                
                if gps_latitude and gps_longitude:
                    lat = convert_to_degrees(gps_latitude)
                    lon = convert_to_degrees(gps_longitude)
                    
                    # Apply hemisphere corrections
                    if gps_latitude_ref and gps_latitude_ref.upper() == 'S':
                        lat = -lat
                    if gps_longitude_ref and gps_longitude_ref.upper() == 'W':
                        lon = -lon
                        
                    metadata['latitude'] = lat
                    metadata['longitude'] = lon
                    logger.info(f"GPS coordinates for {os.path.basename(image_path)}: {lat:.6f}, {lon:.6f}")
                
                if gps_altitude:
                    metadata['altitude'] = float(gps_altitude)
            else:
                logger.warning(f"No GPS data found in {os.path.basename(image_path)}")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {image_path}: {str(e)}")
            return {
                'filename': os.path.basename(image_path),
                'timestamp': None,
                'latitude': None,
                'longitude': None,
                'altitude': None
            }
    
    def process_image_with_workflow(self, image_path: str) -> tuple[Dict[str, Any], Optional[str]]:
        """Process single image through Roboflow workflow and return detections + output image path"""
        try:
            # Try to connect to inference server first
            try:
                client = InferenceHTTPClient(
                    api_url="http://localhost:9001",  # local inference server
                    api_key=ROBOFLOW_API_KEY
                )
                
                # Test connection
                import requests
                response = requests.get("http://localhost:9001/", timeout=2)
                
                # Run workflow if server is available
                result = client.run_workflow(
                    workspace_name=ROBOFLOW_WORKSPACE,
                    workflow_id=ROBOFLOW_WORKFLOW_ID,
                    images={
                        "image": image_path
                    }
                )
                
                # Save output image if available in workflow result
                output_image_path = None
                try:
                    # Check for different possible output image keys (similar to video processing)
                    output_image_data = None
                    output_key = None
                    
                    # Handle both dict and list results
                    result_dict = None
                    if isinstance(result, dict):
                        result_dict = result
                    elif isinstance(result, list) and len(result) > 0:
                        result_dict = result[0]  # Take first item from list
                        logger.info(f"Workflow returned list with {len(result)} items, using first item")
                    
                    if result_dict:
                        # Check for various output image keys in priority order
                        for key in ["output_image", "label_visualization", "car_model_predictions", "license_plate_model_predictions", "fallback_visual"]:
                            if result_dict.get(key):
                                output_image_data = result_dict[key]
                                output_key = key
                                break
                    
                    if output_image_data:
                        logger.info(f"Found output image data in key '{output_key}': type={type(output_image_data)}")
                        logger.info(f"Output image data attributes: {dir(output_image_data)}")
                        
                        # Create output directory
                        output_dir = os.path.join(self.temp_dir, 'outputs')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save the output image
                        output_image_path = os.path.join(output_dir, f"processed_{os.path.basename(image_path)}")
                        
                        # Handle different types of output image data
                        if hasattr(output_image_data, 'numpy_image'):
                            # If it has numpy_image attribute, use cv2 to save
                            import cv2
                            logger.info(f"Using {output_key}.numpy_image (shape: {output_image_data.numpy_image.shape})")
                            cv2.imwrite(output_image_path, output_image_data.numpy_image)
                            logger.info(f"✅ Saved output image using {output_key}.numpy_image: {output_image_path}")
                        elif hasattr(output_image_data, 'save'):
                            # If it's a PIL image
                            logger.info(f"Using {output_key}.save() method")
                            output_image_data.save(output_image_path)
                            logger.info(f"✅ Saved output image using {output_key}.save(): {output_image_path}")
                        elif isinstance(output_image_data, str):
                            # If it's a base64 encoded string
                            logger.info(f"Using base64 string data for {output_key}")
                            try:
                                import base64
                                from PIL import Image
                                import io
                                
                                # Handle data URL format
                                if output_image_data.startswith('data:image/'):
                                    # Remove data URL prefix
                                    base64_data = output_image_data.split(',')[1]
                                else:
                                    base64_data = output_image_data
                                
                                # Decode base64
                                image_bytes = base64.b64decode(base64_data)
                                
                                # Open with PIL and save
                                image = Image.open(io.BytesIO(image_bytes))
                                image.save(output_image_path)
                                logger.info(f"✅ Saved base64 output image using {output_key}: {output_image_path} (size: {image.size})")
                                
                            except Exception as base64_error:
                                logger.warning(f"❌ Failed to decode base64 image for {output_key}: {base64_error}")
                                output_image_path = None
                        else:
                            logger.warning(f"❌ Unknown output_image format for {output_key}: {type(output_image_data)}")
                            logger.warning(f"Available attributes: {[attr for attr in dir(output_image_data) if not attr.startswith('_')]}")
                            output_image_path = None
                    else:
                        logger.info("No output image data found in workflow result")
                            
                except Exception as img_error:
                    logger.warning(f"Could not save output image: {str(img_error)}")
                    output_image_path = None
                
                logger.info(f"Successfully processed image with workflow: {os.path.basename(image_path)}")
                return self.make_serializable(result), output_image_path
                
            except (requests.exceptions.RequestException, Exception) as server_error:
                logger.warning(f"Inference server not available ({str(server_error)}), using placeholder detection data")
                
                # Return placeholder detection data for testing
                placeholder_result = [
                        {
                            "open_ai": [],
                            "output_image": "<excluded_image_data_str>",
                            "fallback_visual": "<excluded_image_data_str>",
                            "trash_model_predictions": {
                                "image": {
                                    "width": None,
                                    "height": None
                                },
                                "predictions": []
                            },
                            "car_model_predictions": {
                            "image": {
                                "width": None,
                                "height": None
                            },
                            "predictions": []
                            },
                            "license_plate_model_predictions": {
                            "image": {
                                "width": None,
                                "height": None
                            },
                            "predictions": []
                            }
                        }
                    ],
                
                return placeholder_result, None
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {"error": str(e)}, None
    
    def make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, "__dict__"):
            filtered_dict = {}
            for k, v in obj.__dict__.items():
                # Skip image data attributes
                if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image', 'output_image', 'fallback_visual']:
                    filtered_dict[k] = f"<excluded_image_data_{type(v).__name__}>"
                else:
                    filtered_dict[k] = self.make_serializable(v)
            return filtered_dict
        elif isinstance(obj, (list, tuple)):
            return [self.make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            # Filter out image data from dictionaries too
            filtered_dict = {}
            for k, v in obj.items():
                if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image', 'output_image', 'fallback_visual']:
                    filtered_dict[k] = f"<excluded_image_data_{type(v).__name__}>"
                else:
                    filtered_dict[k] = self.make_serializable(v)
            return filtered_dict
        elif hasattr(obj, 'tolist'):  # numpy arrays
            # For numpy arrays, check if it's likely image data (large arrays)
            if obj.size > 1000:  # Skip large arrays (likely images)
                return f"<excluded_large_array_shape_{obj.shape}>"
            return obj.tolist()
        else:
            return obj
    
    def add_logo_overlay(self, image_path: str) -> str:
        """Add company logo overlay to image with 50% opacity in top right corner"""
        try:
            # Load the main image
            main_image = Image.open(image_path)
            
            # Check if the image was saved by cv2 (BGR format) and convert to RGB
            if main_image.mode == 'RGB':
                # Convert BGR to RGB if the image was saved by cv2
                import cv2
                import numpy as np
                
                # Load with cv2 to get BGR format
                cv2_image = cv2.imread(image_path)
                if cv2_image is not None:
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                    # Convert numpy array to PIL Image
                    main_image = Image.fromarray(rgb_image)
                    logger.info(f"✅ Converted BGR image to RGB for logo overlay: {os.path.basename(image_path)}")
            
            # Load the company logo (favicon.png)
            logo_path = os.path.join(os.path.dirname(__file__), 'favicon.png')
            if not os.path.exists(logo_path):
                logger.warning(f"Logo file not found at {logo_path}, skipping logo overlay")
                return image_path
            
            logo = Image.open(logo_path)
            
            # Convert logo to RGBA if it isn't already
            if logo.mode != 'RGBA':
                logo = logo.convert('RGBA')
            
            # Store original mode for later conversion
            original_mode = main_image.mode
            
            # Convert main image to RGBA for compositing
            if main_image.mode != 'RGBA':
                main_image = main_image.convert('RGBA')
            
            # Calculate logo size (10% of image width, maintaining aspect ratio)
            logo_width = int(main_image.width * 0.1)
            logo_height = int((logo.height * logo_width) / logo.width)
            logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            
            # Apply 50% opacity to logo
            logo_with_alpha = Image.new('RGBA', logo.size, (0, 0, 0, 0))
            for x in range(logo.width):
                for y in range(logo.height):
                    r, g, b, a = logo.getpixel((x, y))
                    # Apply 50% opacity (multiply alpha by 0.5)
                    new_alpha = int(a * 0.5)
                    logo_with_alpha.putpixel((x, y), (r, g, b, new_alpha))
            
            # Calculate position for top right corner with some padding
            padding = 20
            x_position = main_image.width - logo_width - padding
            y_position = padding
            
            # Create a copy of the main image to avoid modifying the original
            output_image = main_image.copy()
            
            # Paste logo onto the main image
            output_image.paste(logo_with_alpha, (x_position, y_position), logo_with_alpha)
            
            # Convert back to original mode before saving
            if original_mode != 'RGBA':
                output_image = output_image.convert(original_mode)
            
            # Save the image with logo overlay
            # If the original image was saved by cv2, we need to convert back to BGR for cv2 compatibility
            if original_mode == 'RGB':
                # Convert RGB back to BGR for cv2 compatibility
                import cv2
                import numpy as np
                
                # Convert PIL to numpy array (RGB)
                rgb_array = np.array(output_image)
                # Convert RGB to BGR
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                # Save with cv2 to maintain BGR format
                cv2.imwrite(image_path, bgr_array)
                logger.info(f"✅ Saved image with logo overlay in BGR format: {os.path.basename(image_path)}")
            else:
                # Save with PIL for other formats
                output_image.save(image_path)
            logger.info(f"✅ Added logo overlay to image: {os.path.basename(image_path)}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error adding logo overlay to {image_path}: {str(e)}")
            return image_path  # Return original path if overlay fails
    
    def upload_output_image_to_s3(self, local_image_path: str, filename: str) -> str:
        """Upload processed image to S3 outputs folder and return S3 URL"""
        try:
            s3_client = boto3.client('s3')
            
            # Parse output folder URL to get bucket and key prefix
            # s3://bucket/folder/path/outputs/ -> bucket: "bucket", key_prefix: "folder/path/outputs/"
            output_parts = self.s3_output_folder_url.replace('s3://', '').split('/', 1)
            bucket = output_parts[0]
            key_prefix = output_parts[1] if len(output_parts) > 1 else ''
            
            # Create S3 key for output image
            s3_key = f"{key_prefix}{filename}"
            
            # Upload image
            s3_client.upload_file(local_image_path, bucket, s3_key)
            
            # Create S3 URL
            s3_url = f"s3://{bucket}/{s3_key}"
            
            return s3_url
            
        except Exception as e:
            logger.error(f"Error uploading image to S3: {str(e)}")
            raise
    
    def store_in_database(self, image_data: Dict[str, Any]):
        """Store processed image data in Supabase"""
        try:
            # Insert into images table
            result = supabase.table('processed_images').insert({
                'task_id': self.task_id,
                'filename': image_data['metadata']['filename'],
                's3_input_url': image_data.get('s3_input_url'),
                's3_output_url': image_data.get('s3_output_url'),
                'timestamp': image_data['metadata']['timestamp'],
                'latitude': image_data['metadata']['latitude'],
                'longitude': image_data['metadata']['longitude'],
                'altitude': image_data['metadata']['altitude'],
                'detections': self.make_serializable(image_data['detections']),
                'processing_status': image_data.get('processing_status', 'success'),
                'error_message': image_data.get('error_message'),
                'processed_at': datetime.now().isoformat(),
            }).execute()
            
            return result.data[0]['id'] if result.data else None
            
        except Exception as e:
            logger.error(f"Error storing data in database: {str(e)}")
            raise
    
    def generate_folder_name_from_timestamp(self, timestamp_str: str) -> str:
        """Generate folder name from timestamp in format 'Run [Month] [Date] [time in 12 hour]'"""
        try:
            # Parse the ISO timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Format: Run [Month] [Date] [time in 12 hour]
            # Example: "Run January 15 2:30 PM"
            month_name = dt.strftime('%B')  # Full month name
            day = dt.strftime('%d').lstrip('0')  # Day without leading zero
            time_12hr = dt.strftime('%I:%M %p').lstrip('0')  # 12-hour time without leading zero
            
            folder_name = f"Run {month_name} {day} {time_12hr}"
            return folder_name
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing timestamp for folder name: {e}")
            # Fallback to original naming if timestamp parsing fails
            return f"Run {datetime.now().strftime('%B %d %I:%M %p')}"
    
    
    def create_folder_record(self, s3_folder_url: str, folder_name: str):
        """Create folder processing record in database"""
        try:
            result = supabase.table('processed_folders').insert({
                'task_id': self.task_id,
                'folder_name': folder_name,
                's3_input_folder_url': s3_folder_url,
                's3_output_folder_url': self.s3_output_folder_url,
                'status': 'processing',
                'processing_started_at': datetime.now().isoformat()
            }).execute()
            
            return result.data[0]['id'] if result.data else None
            
        except Exception as e:
            logger.error(f"Error creating folder record: {str(e)}")
            raise
    
    def update_folder_record(self, status: str, error_message: str = None):
        """Update folder processing record"""
        try:
            update_data = {
                'status': status,
                'total_images': self.total_count,
                'processed_images': self.processed_count,
                'successful_images': self.successful_count,
                'failed_images': self.failed_count,
                'updated_at': datetime.now().isoformat()
            }
            
            if status == 'completed':
                update_data['processing_completed_at'] = datetime.now().isoformat()
            
            if error_message:
                update_data['error_message'] = error_message
            
            supabase.table('processed_folders').update(update_data).eq('task_id', self.task_id).execute()
            
        except Exception as e:
            logger.error(f"Error updating folder record: {str(e)}")
            raise
    
    def process_folder(self, s3_folder_url: str):
        """Main processing function"""
        try:
            # Update task status
            tasks[self.task_id]["status"] = "processing"
            tasks[self.task_id]["progress"] = "Initializing..."
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix=f"image_process_{self.task_id}_")
            
            # Download images first to get metadata for folder naming
            tasks[self.task_id]["progress"] = "Downloading images..."
            image_paths = self.download_images_from_s3_folder(s3_folder_url)
            tasks[self.task_id]["total_images"] = len(image_paths)
            self.total_count = len(image_paths)
            
            # Generate folder name from first image timestamp
            folder_name = s3_folder_url.rstrip('/').split('/')[-1]  # Default fallback
            if image_paths:
                try:
                    # Sort images by filename (assuming timestamp-based naming) and get first one
                    sorted_paths = sorted(image_paths)
                    first_image_metadata = self.extract_image_metadata(sorted_paths[0])
                    if first_image_metadata.get('timestamp'):
                        folder_name = self.generate_folder_name_from_timestamp(first_image_metadata['timestamp'])
                        logger.info(f"Generated folder name: {folder_name}")
                    else:
                        logger.warning("No timestamp found in first image, using default folder name")
                except Exception as e:
                    logger.warning(f"Error generating folder name from first image: {e}")
            
            # Create folder record in database
            self.create_folder_record(s3_folder_url, folder_name)
            
            # Process each image
            for i, image_path in enumerate(image_paths):
                try:
                    tasks[self.task_id]["progress"] = f"Processing image {i+1}/{len(image_paths)}"
                    
                    # Extract metadata
                    metadata = self.extract_image_metadata(image_path)
                    
                    # Run through workflow
                    detections, output_image_path = self.process_image_with_workflow(image_path)
                    
                    # Add logo overlay to output image if available
                    if output_image_path and os.path.exists(output_image_path):
                        output_image_path = self.add_logo_overlay(output_image_path)
                    
                    # Upload output image to S3 if available
                    s3_output_url = None
                    if output_image_path and os.path.exists(output_image_path):
                        s3_output_url = self.upload_output_image_to_s3(
                            output_image_path, 
                            f"processed_{metadata['filename']}"
                        )
                    
                    # Generate correct S3 input URL
                    # Extract bucket and folder from original S3 URL
                    s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
                    bucket_name = s3_parts[0]
                    folder_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
                    correct_s3_input_url = f"s3://{bucket_name}/{folder_prefix}/{metadata['filename']}"
                    
                    # Store in database
                    image_data = {
                        'metadata': metadata,
                        'detections': detections,
                        's3_input_url': correct_s3_input_url,
                        's3_output_url': s3_output_url,
                        'processing_status': 'success'
                    }
                    self.store_in_database(image_data)
                    
                    self.processed_count += 1
                    self.successful_count += 1
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {image_path}: {str(img_error)}")
                    
                    # Store failed image record
                    metadata = self.extract_image_metadata(image_path)
                    image_data = {
                        'metadata': metadata,
                        'detections': {},
                        'processing_status': 'failed',
                        'error_message': str(img_error)
                    }
                    self.store_in_database(image_data)
                    
                    self.processed_count += 1
                    self.failed_count += 1
                
                # Update task progress
                tasks[self.task_id]["processed_images"] = self.processed_count
                tasks[self.task_id]["successful_images"] = self.successful_count
                tasks[self.task_id]["failed_images"] = self.failed_count
                tasks[self.task_id]["s3_output_folder_url"] = self.s3_output_folder_url
                
                logger.info(f"Processed image {i+1}/{len(image_paths)} for task {self.task_id}")
            
            # Update final status
            tasks[self.task_id]["status"] = "completed"
            tasks[self.task_id]["progress"] = f"Completed! Processed {self.processed_count} images ({self.successful_count} successful, {self.failed_count} failed)"
            
            # Update folder record
            self.update_folder_record('completed')
            
            logger.info(f"Task {self.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing folder for task {self.task_id}: {str(e)}")
            tasks[self.task_id]["status"] = "failed"
            tasks[self.task_id]["error"] = str(e)
            
            # Update folder record with error
            try:
                self.update_folder_record('failed', str(e))
            except:
                pass
                
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

def count_images_in_s3_folder(s3_folder_url: str) -> int:
    """Count the number of image files in an S3 folder"""
    try:
        # Parse S3 URL to get bucket and prefix
        s3_parts = s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
        bucket = s3_parts[0]
        folder_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
        
        # List all objects in the folder
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=folder_prefix
        )
        
        if 'Contents' not in response:
            return 0
        
        # Count image files (only in the specific folder, not subfolders)
        image_count = 0
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip if this is a subfolder (contains additional path separators after the folder prefix)
            # Handle folder prefix with or without trailing slash
            if folder_prefix.endswith('/'):
                relative_path = key[len(folder_prefix):]
            else:
                relative_path = key[len(folder_prefix + '/'):] if key.startswith(folder_prefix + '/') else key[len(folder_prefix):]
            
            # Skip if this is in a subfolder (relative path contains slashes)
            if '/' in relative_path:
                continue  # Skip subfolders
            
            # Count only image files directly in this folder
            if key.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                image_count += 1
        
        return image_count
        
    except Exception as e:
        logger.warning(f"Error counting images in S3 folder {s3_folder_url}: {e}")
        return 0

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

def generate_folder_name_from_timestamp_standalone(timestamp_str: str) -> str:
    """Generate folder name from timestamp in format 'Run [Month] [Date] [time in 12 hour]'"""
    try:
        # Parse the ISO timestamp
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Format: Run [Month] [Date] [time in 12 hour]
        # Example: "Run January 15 2:30 PM"
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
            
        metadata = {
            'filename': os.path.basename(image_path),
            'timestamp': None,
            'latitude': None,
            'longitude': None,
            'altitude': None
        }
        
        if not exif_dict:
            logger.warning(f"No EXIF data found in {os.path.basename(image_path)}")
            return metadata
        
        # Extract timestamp
        for tag_id, value in exif_dict.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                try:
                    metadata['timestamp'] = datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
                    break  # Use first valid timestamp found
                except ValueError:
                    continue
        
        # Extract GPS data
        gps_ifd = exif_dict.get_ifd(0x8825)  # GPS IFD tag
        if gps_ifd:
            logger.info(f"Found GPS data in {os.path.basename(image_path)}")
            
            def convert_to_degrees(value):
                """Convert GPS coordinates from degrees/minutes/seconds to decimal degrees"""
                if isinstance(value, (tuple, list)) and len(value) == 3:
                    d, m, s = value
                    return float(d) + float(m)/60 + float(s)/3600
                return 0.0
            
            # Extract GPS coordinates
            gps_latitude = gps_ifd.get(2)  # GPSLatitude
            gps_latitude_ref = gps_ifd.get(1)  # GPSLatitudeRef
            gps_longitude = gps_ifd.get(4)  # GPSLongitude  
            gps_longitude_ref = gps_ifd.get(3)  # GPSLongitudeRef
            gps_altitude = gps_ifd.get(6)  # GPSAltitude
            
            if gps_latitude and gps_longitude:
                lat = convert_to_degrees(gps_latitude)
                lon = convert_to_degrees(gps_longitude)
                
                # Apply hemisphere corrections
                if gps_latitude_ref and gps_latitude_ref.upper() == 'S':
                    lat = -lat
                if gps_longitude_ref and gps_longitude_ref.upper() == 'W':
                    lon = -lon
                    
                metadata['latitude'] = lat
                metadata['longitude'] = lon
                logger.info(f"GPS coordinates for {os.path.basename(image_path)}: {lat:.6f}, {lon:.6f}")
                
                if gps_altitude:
                    metadata['altitude'] = float(gps_altitude)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {str(e)}")
        return {
            'filename': os.path.basename(image_path),
            'timestamp': None,
            'latitude': None,
            'longitude': None,
            'altitude': None
        }

# API Endpoints
@app.post("/process-folder", response_model=ImageProcessResponse)
async def process_folder_endpoint(request: ImageFolderProcessRequest, background_tasks: BackgroundTasks):
    """Process images from an S3 folder"""
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task
        tasks[task_id] = {
            "status": "queued",
            "progress": None,
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_images": None,
            "s3_output_folder_url": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Create processor and start background task
        processor = ImageProcessor(task_id, request.s3_folder_url)
        background_tasks.add_task(processor.process_folder, request.s3_folder_url)
        
        return ImageProcessResponse(
            task_id=task_id,
            status="queued",
            message="Image processing started. Use /status/{task_id} to check progress."
        )
        
    except Exception as e:
        logger.error(f"Error starting image processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerun-folder", response_model=ImageProcessResponse)
async def rerun_folder_endpoint(request: ImageFolderProcessRequest, background_tasks: BackgroundTasks):
    """Rerun processing for a folder (useful when initial processing fails)"""
    try:
        # Check if folder exists in S3
        s3_parts = request.s3_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
        bucket = s3_parts[0]
        folder_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
        
        s3_client = boto3.client('s3')
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=folder_prefix,
                MaxKeys=1
            )
            if 'Contents' not in response:
                raise HTTPException(status_code=404, detail=f"S3 folder not found: {request.s3_folder_url}")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Error accessing S3 folder: {str(e)}")
        
        # Check if folder was previously processed
        existing_folders = supabase.table('processed_folders').select('*').eq('s3_input_folder_url', request.s3_folder_url).execute()
        
        # Clean up previous processing data if it exists
        if existing_folders.data:
            logger.info(f"Cleaning up previous processing data for folder: {request.s3_folder_url}")
            
            # Get the task_id from previous processing
            old_task_id = existing_folders.data[0]['task_id']
            
            # Delete processed images
            supabase.table('processed_images').delete().eq('task_id', old_task_id).execute()
            
            # Delete processed folder record
            supabase.table('processed_folders').delete().eq('s3_input_folder_url', request.s3_folder_url).execute()
            
            # Remove from tasks if still in memory
            if old_task_id in tasks:
                del tasks[old_task_id]
            
            logger.info(f"Cleaned up previous processing data for task_id: {old_task_id}")
        
        # Generate new unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize new task
        tasks[task_id] = {
            "status": "queued",
            "progress": None,
            "processed_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_images": None,
            "s3_output_folder_url": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Create processor and start background task
        processor = ImageProcessor(task_id, request.s3_folder_url)
        background_tasks.add_task(processor.process_folder, request.s3_folder_url)
        
        logger.info(f"Rerun processing started for folder: {request.s3_folder_url} with task_id: {task_id}")
        
        return ImageProcessResponse(
            task_id=task_id,
            status="queued",
            message=f"Rerun processing started for folder. Use /status/{task_id} to check progress."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting rerun processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting rerun processing: {str(e)}")

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of an image processing task"""
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
        s3_output_folder_url=task_data["s3_output_folder_url"],
        error=task_data["error"]
    )

@app.get("/")
async def root():
    """API health check"""
    return {"message": "Image Processing API is running", "version": "1.0.0"}

@app.get("/folders")
async def get_processed_folders():
    """Get list of all processed folders sorted by timestamp (oldest first)"""
    try:
        result = supabase.table('processed_folders').select('*').execute()
        
        # Sort by timestamp extracted from folder names (oldest first)
        def get_folder_timestamp(folder_info):
            try:
                folder_name = folder_info.get('folder_name', '')
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
                            from datetime import datetime
                            current_year = datetime.now().year
                            dt_str = f"{current_year} {month_name} {day} {time_part}"
                            dt = datetime.strptime(dt_str, "%Y %B %d %I:%M %p")
                            return dt
                    except Exception as e:
                        logger.warning(f"Error parsing folder name timestamp '{folder_name}': {e}")
                
                # Fallback: return epoch time (will sort to beginning)
                from datetime import datetime
                return datetime(1970, 1, 1)
                
            except Exception as e:
                logger.warning(f"Error extracting timestamp for folder: {e}")
                from datetime import datetime
                return datetime(1970, 1, 1)
        
        # Sort by timestamp (oldest first)
        sorted_folders = sorted(result.data, key=get_folder_timestamp, reverse=True)
        
        return {"folders": sorted_folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching folders: {str(e)}")

@app.get("/folders/{task_id}/images")
async def get_folder_images(task_id: str):
    """Get all processed images for a specific folder"""
    try:
        result = supabase.table('processed_images').select('*').eq('task_id', task_id).order('timestamp').execute()
        images_with_signed_urls = add_signed_urls_to_images(result.data)
        return {"images": images_with_signed_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching images: {str(e)}")

@app.post("/compare-folders", response_model=ImageCompareResponse)
async def compare_folders(request: ImageCompareRequest):
    """
    Compare images from two folders based on GPS coordinates and find matches within a distance threshold.
    Returns matched image pairs with their detections.
    """
    try:
        # Fetch images from both folders
        folder1_result = supabase.table('processed_images').select('*').eq('task_id', request.folder1_task_id).order('timestamp').execute()
        folder2_result = supabase.table('processed_images').select('*').eq('task_id', request.folder2_task_id).order('timestamp').execute()
        
        folder1_images = folder1_result.data
        folder2_images = folder2_result.data
        
        if not folder1_images:
            raise HTTPException(status_code=404, detail=f"No images found for folder1 task_id: {request.folder1_task_id}")
        if not folder2_images:
            raise HTTPException(status_code=404, detail=f"No images found for folder2 task_id: {request.folder2_task_id}")
        
        logger.info(f"Comparing {len(folder1_images)} images from folder1 with {len(folder2_images)} images from folder2")
        
        # Haversine formula to calculate distance between two GPS coordinates
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            Returns distance in meters
            """
            if None in [lat1, lon1, lat2, lon2]:
                return float('inf')
            
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Radius of earth in meters
            r = 6371000
            
            return c * r
        
        # Filter out images without valid GPS coordinates
        folder1_valid = [img for img in folder1_images if img.get('latitude') is not None and img.get('longitude') is not None]
        folder2_valid = [img for img in folder2_images if img.get('latitude') is not None and img.get('longitude') is not None]
        
        logger.info(f"Valid GPS coordinates: folder1={len(folder1_valid)}, folder2={len(folder2_valid)}")
        
        matches = []
        distance_threshold = request.distance_threshold_meters
        
        # Find matches within distance threshold
        for img1 in folder1_valid:
            lat1 = float(img1['latitude'])
            lon1 = float(img1['longitude'])
            
            for img2 in folder2_valid:
                lat2 = float(img2['latitude'])
                lon2 = float(img2['longitude'])
                
                # Calculate distance between GPS coordinates
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                if distance <= distance_threshold:
                    match = ImageMatch(
                        image1=img1,
                        image2=img2,
                        distance_meters=round(distance, 2)
                    )
                    matches.append(match)
        
        # Add signed URLs to all matched images
        for match in matches:
            match.image1 = add_signed_urls_to_images([match.image1])[0]
            match.image2 = add_signed_urls_to_images([match.image2])[0]
        
        # Sort matches by distance (closest matches first)
        matches.sort(key=lambda x: x.distance_meters)
        
        logger.info(f"Found {len(matches)} matches within {request.distance_threshold_meters}m threshold")
        
        return ImageCompareResponse(
            matches=matches,
            total_matches=len(matches),
            folder1_images=len(folder1_images),
            folder2_images=len(folder2_images),
            distance_threshold_meters=request.distance_threshold_meters
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.post("/compare-folders-enriched", response_model=EnrichedCompareResponse)
async def compare_folders_enriched(request: ImageCompareRequest):
    """
    Enhanced folder comparison that analyzes detection changes between matched images.
    Perfect for change detection: cars leaving trash, objects appearing/disappearing, etc.
    """
    try:
        # Fetch images from both folders
        folder1_result = supabase.table('processed_images').select('*').eq('task_id', request.folder1_task_id).order('timestamp').execute()
        folder2_result = supabase.table('processed_images').select('*').eq('task_id', request.folder2_task_id).order('timestamp').execute()
        
        folder1_images = folder1_result.data
        folder2_images = folder2_result.data
        
        if not folder1_images:
            raise HTTPException(status_code=404, detail=f"No images found for folder1 task_id: {request.folder1_task_id}")
        if not folder2_images:
            raise HTTPException(status_code=404, detail=f"No images found for folder2 task_id: {request.folder2_task_id}")
        
        logger.info(f"Enriched comparison: {len(folder1_images)} vs {len(folder2_images)} images")
        
        # Haversine distance calculation
        import math
        from datetime import datetime
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c
        
        # Filter images with valid GPS
        folder1_valid = [img for img in folder1_images if img.get('latitude') and img.get('longitude')]
        folder2_valid = [img for img in folder2_images if img.get('latitude') and img.get('longitude')]
        
        enriched_matches = []
        change_stats = {
            'objects_appeared': 0,
            'objects_disappeared': 0, 
            'count_changes': 0,
            'no_changes': 0
        }
        
        # Find matches and analyze changes
        for img1 in folder1_valid:
            lat1 = float(img1['latitude'])
            lon1 = float(img1['longitude'])
            
            for img2 in folder2_valid:
                lat2 = float(img2['latitude']) 
                lon2 = float(img2['longitude'])
                
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                if distance <= request.distance_threshold_meters:
                    # Calculate time difference
                    time_diff = 0
                    try:
                        if img1.get('timestamp') and img2.get('timestamp'):
                            t1 = datetime.fromisoformat(img1['timestamp'].replace('Z', '+00:00'))
                            t2 = datetime.fromisoformat(img2['timestamp'].replace('Z', '+00:00'))
                            time_diff = abs((t2 - t1).total_seconds() / 60)  # minutes
                    except:
                        pass
                    
                    # Analyze detection changes
                    changes, summary = analyze_detection_changes(img1, img2)
                    
                    # Update statistics
                    if not changes:
                        change_stats['no_changes'] += 1
                    else:
                        for change in changes:
                            if change.change_type == 'appeared':
                                change_stats['objects_appeared'] += 1
                            elif change.change_type == 'disappeared':
                                change_stats['objects_disappeared'] += 1
                            elif change.change_type == 'count_changed':
                                change_stats['count_changes'] += 1
                    
                    # Add signed URLs
                    img1_with_urls = add_signed_urls_to_images([img1])[0]
                    img2_with_urls = add_signed_urls_to_images([img2])[0]
                    
                    enriched_match = EnrichedImageMatch(
                        image1=img1_with_urls,
                        image2=img2_with_urls,
                        distance_meters=round(distance, 2),
                        time_difference_minutes=round(time_diff, 1),
                        detection_changes=changes,
                        change_summary=summary
                    )
                    enriched_matches.append(enriched_match)
        
        # Sort by distance (closest first)
        enriched_matches.sort(key=lambda x: x.distance_meters)
        
        logger.info(f"Found {len(enriched_matches)} enriched matches with change analysis")
        
        return EnrichedCompareResponse(
            matches=enriched_matches,
            total_matches=len(enriched_matches),
            folder1_images=len(folder1_images),
            folder2_images=len(folder2_images),
            distance_threshold_meters=request.distance_threshold_meters,
            change_statistics=change_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enriched comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enriched comparison error: {str(e)}")

@app.post("/analyze-multifolder-dumping", response_model=MultifolderAnalysisResponse)
async def analyze_multifolder_dumping(request: MultifolderAnalysisRequest):
    """
    Analyze multiple folders for car dumping violations across time.
    Clusters images by GPS location, sorts by timestamp, and detects dumping patterns.
    """
    try:
        import math
        from datetime import datetime
        from collections import defaultdict
        
        logger.info(f"Starting multi-folder analysis for {len(request.task_ids)} folders")
        
        # Collect all images from all folders
        all_images = []
        folder_names = []
        
        for task_id in request.task_ids:
            # Get images from this folder
            result = supabase.table('processed_images').select('*').eq('task_id', task_id).execute()
            images = result.data
            
            if not images:
                logger.warning(f"No images found for task_id: {task_id}")
                continue
                
            # Get folder name
            folder_result = supabase.table('processed_folders').select('folder_name').eq('task_id', task_id).execute()
            folder_name = folder_result.data[0]['folder_name'] if folder_result.data else task_id
            folder_names.append(folder_name)
            
            # Add folder info to each image and filter valid GPS
            for img in images:
                if img.get('latitude') and img.get('longitude') and img.get('timestamp'):
                    img['folder_name'] = folder_name
                    img['task_id'] = task_id
                    all_images.append(img)
        
        logger.info(f"Collected {len(all_images)} images with GPS from {len(folder_names)} folders")
        
        if len(all_images) < 2:
            return MultifolderAnalysisResponse(
                violations=[],
                total_violations=0,
                clusters_analyzed=0,
                total_frames=len(all_images),
                folders_analyzed=folder_names,
                analysis_summary={}
            )
        
        # Haversine distance function
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c
        
        # Spatial clustering - group images by GPS location
        clusters = []
        cluster_id = 0
        
        for img in all_images:
            lat, lon = float(img['latitude']), float(img['longitude'])
            
            # Find existing cluster within threshold
            assigned_cluster = None
            for cluster in clusters:
                cluster_center_lat = sum(float(i['latitude']) for i in cluster['images']) / len(cluster['images'])
                cluster_center_lon = sum(float(i['longitude']) for i in cluster['images']) / len(cluster['images'])
                
                distance = haversine_distance(lat, lon, cluster_center_lat, cluster_center_lon)
                if distance <= request.distance_threshold_meters:
                    assigned_cluster = cluster
                    break
            
            if assigned_cluster:
                assigned_cluster['images'].append(img)
            else:
                # Create new cluster
                clusters.append({
                    'id': cluster_id,
                    'images': [img]
                })
                cluster_id += 1
        
        logger.info(f"Created {len(clusters)} spatial clusters")
        
        # Analyze each cluster for dumping violations
        violations = []
        analysis_summary = defaultdict(int)
        
        for cluster in clusters:
            cluster_images = cluster['images']
            
            # Sort images by timestamp
            cluster_images.sort(key=lambda x: datetime.fromisoformat(x['timestamp'].replace('Z', '+00:00')))
            
            # Calculate cluster center
            cluster_lat = sum(float(img['latitude']) for img in cluster_images) / len(cluster_images)
            cluster_lon = sum(float(img['longitude']) for img in cluster_images) / len(cluster_images)
            
            # Look for dumping patterns across time
            for i in range(len(cluster_images) - 1):
                for j in range(i + 1, len(cluster_images)):
                    img1 = cluster_images[i]  # Earlier image
                    img2 = cluster_images[j]  # Later image
                    
                    # Check time gap
                    time1 = datetime.fromisoformat(img1['timestamp'].replace('Z', '+00:00'))
                    time2 = datetime.fromisoformat(img2['timestamp'].replace('Z', '+00:00'))
                    time_diff = (time2 - time1).total_seconds() / 60  # minutes
                    
                    if time_diff < request.min_time_gap_minutes:
                        continue  # Too close in time
                    
                    # Analyze dumping pattern
                    changes, summary = analyze_detection_changes(img1, img2)
                    
                    if "DUMPING DETECTED" in summary:
                        # Extract violation details
                        violation_type = "unknown"
                        trash_count = 0
                        
                        for change in changes:
                            if change.object_type == "dumping_violation":
                                violation_type = change.change_type
                                trash_count = change.after_count
                                break
                        
                        # Extract license plate from before frame if available
                        license_plate = None
                        try:
                            detections = img1.get('detections', [])
                            if detections and len(detections) > 0:
                                detection = detections[0]
                                if isinstance(detection, list) and len(detection) > 0:
                                    detection = detection[0]
                                
                                # Check OpenAI detections for license plate
                                openai_data = detection.get('open_ai', [])
                                if openai_data and len(openai_data) > 0:
                                    potential_plate = openai_data[0]  # First entry is usually license plate
                                    # Filter out OpenAI's "unable to read" responses and generic placeholders
                                    if potential_plate and not any(phrase in potential_plate.lower() for phrase in ["unable to read", "can't read", "cannot read", "transcribe license", "license_plate", "license plate", "UNREADABLE", "[UNREADABLE.]", "UNREADABLE", "UNREADABLE.", "unreadable", "unreadable.", "not readable", "cannot be read", "unclear", "blurry", "not visible", "no license plate"]):
                                        license_plate = potential_plate
                                
                                # Also check license plate model predictions
                                if not license_plate:
                                    plate_preds = detection.get('license_plate_model_predictions', {}).get('predictions', [])
                                    if plate_preds and len(plate_preds) > 0:
                                        # Extract text from license plate prediction
                                        plate_data = plate_preds[0]
                                        if isinstance(plate_data, dict) and 'class' in plate_data:
                                            potential_plate = plate_data['class']
                                            # Filter out generic placeholders from model predictions too
                                            if potential_plate and potential_plate.lower() not in ["license_plate", "license plate", "unknown", "n/a"]:
                                                license_plate = potential_plate
                        except Exception as e:
                            logger.warning(f"Error extracting license plate: {str(e)}")
                        
                        # Update description with license plate if available
                        if license_plate:
                            # Clean up license plate text (remove extra spaces, etc.)
                            license_plate = license_plate.strip()
                            if violation_type == "car_left_trash":
                                updated_description = f"Car [{license_plate}] left {trash_count} trash item(s) behind"
                            elif violation_type == "car_dumped_trash":
                                updated_description = f"Car [{license_plate}] dumped {trash_count} trash item(s)"
                            elif violation_type == "car_arrived_dumped":
                                updated_description = f"Car [{license_plate}] arrived and dumped {trash_count} trash item(s)"
                            elif violation_type == "additional_dumping":
                                updated_description = f"Car [{license_plate}] dumped additional {trash_count} trash item(s)"
                            else:
                                # remove "🚗🗑️ DUMPING DETECTED" from summary
                                updated_description = summary.replace("🚗🗑️ DUMPING DETECTED: ", "")
                                updated_description = summary
                        else:
                            # remove "🚗🗑️ DUMPING DETECTED" from summary
                            updated_description = summary.replace("🚗🗑️ DUMPING DETECTED: ", "")
                        
                        # Add signed URLs
                        img1_with_urls = add_signed_urls_to_images([img1])[0]
                        img2_with_urls = add_signed_urls_to_images([img2])[0]
                        
                        # Calculate distance between before and after GPS coordinates
                        lat1, lon1 = float(img1['latitude']), float(img1['longitude'])
                        lat2, lon2 = float(img2['latitude']), float(img2['longitude'])
                        gps_distance = haversine_distance(lat1, lon1, lat2, lon2)
                        
                        violation = DumpingViolation(
                            cluster_id=cluster['id'],
                            violation_type=violation_type,
                            location={
                                "lat": cluster_lat,
                                "lon": cluster_lon
                            },
                            before_frame=img1_with_urls,
                            after_frame=img2_with_urls,
                            before_folder=img1_with_urls.get('folder_name', 'Unknown'),
                            after_folder=img2_with_urls.get('folder_name', 'Unknown'),
                            vehicle_plate=license_plate,
                            time_difference_minutes=round(time_diff, 1),
                            distance_meters=round(gps_distance, 2),
                            trash_count=trash_count,
                            description=updated_description
                        )
                        
                        # Only count violation if license plate is detected
                        if license_plate:
                            violations.append(violation)
                            analysis_summary[violation_type] += 1
                        # Uncomment below to count violations even without license plate
                        # else:
                        #     violations.append(violation)
                        #     analysis_summary[violation_type] += 1
                        
                        # NOTE: Commented out breaks to find ALL violations in each cluster
                        # Previously this would stop after finding first violation per cluster
                        # Break after finding first violation in this cluster to avoid duplicates
                        break
                    
                # NOTE: Commented out cluster-level break to find ALL violations
                # Previously this would skip to next cluster after finding any violation
                if violations and violations[-1].cluster_id == cluster['id']:
                    break  # Found violation in this cluster, move to next cluster
        
        # Sort violations by time difference (most recent first)
        violations.sort(key=lambda x: x.time_difference_minutes, reverse=True)
        
        logger.info(f"Found {len(violations)} dumping violations across {len(clusters)} clusters")
        
        return MultifolderAnalysisResponse(
            violations=violations,
            total_violations=len(violations),
            clusters_analyzed=len(clusters),
            total_frames=len(all_images),
            folders_analyzed=folder_names,
            analysis_summary=dict(analysis_summary)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-folder dumping analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-folder analysis error: {str(e)}")

@app.post("/folders/{task_id}/generate-srt", response_model=SrtGenerateResponse)
async def generate_srt_for_folder(task_id: str, request: SrtGenerateRequest):
    """
    Generate SRT subtitle file with GPS coordinates for video overlay.
    Perfect for adding GPS data to drone videos at specified frame rates.
    """
    try:
        # Override task_id from URL (more RESTful)
        request.task_id = task_id
        
        # Fetch images from the folder
        images_result = supabase.table('processed_images').select('*').eq('task_id', task_id).order('timestamp').execute()
        images = images_result.data
        
        if not images:
            raise HTTPException(status_code=404, detail=f"No images found for task_id: {task_id}")
        
        logger.info(f"Generating SRT for {len(images)} images, 1s per image")
        
        # Generate SRT content
        srt_content, stats = generate_srt_content(
            images=images,
            font_size=request.font_size,
            precision=request.coordinate_precision
        )
        
        total_entries = stats['images_used']  # Now one entry per image
        
        logger.info(f"Generated SRT with {total_entries} entries from {stats['images_used']} images")
        
        return SrtGenerateResponse(
            srt_content=srt_content,
            total_entries=total_entries,
            duration_seconds=int(stats['total_duration_seconds']),
            gps_range=stats['gps_range'],
            images_used=stats['images_used'],
            interpolated_points=stats['interpolated_points']
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating SRT: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SRT generation error: {str(e)}")


@app.delete("/folders/{task_id}")
async def delete_folder_processing(task_id: str):
    """
    Delete all processing data for a folder:
    - Removes database entries (processed_folders and processed_images)
    - Deletes S3 outputs folder
    - Preserves original folder and its contents
    """
    try:
        # Get folder info first
        folder_result = supabase.table('processed_folders').select('*').eq('task_id', task_id).execute()
        if not folder_result.data:
            raise HTTPException(status_code=404, detail=f"Folder with task_id {task_id} not found")
        
        folder_info = folder_result.data[0]
        s3_input_folder_url = folder_info['s3_input_folder_url']
        s3_output_folder_url = folder_info.get('s3_output_folder_url')
        
        logger.info(f"Deleting processing data for task_id: {task_id}")
        logger.info(f"Input folder: {s3_input_folder_url}")
        logger.info(f"Output folder: {s3_output_folder_url}")
        
        # Delete all processed images from database
        images_result = supabase.table('processed_images').delete().eq('task_id', task_id).execute()
        deleted_images_count = len(images_result.data) if images_result.data else 0
        logger.info(f"Deleted {deleted_images_count} image records from database")
        
        # Delete folder record from database
        folder_delete_result = supabase.table('processed_folders').delete().eq('task_id', task_id).execute()
        deleted_folders_count = len(folder_delete_result.data) if folder_delete_result.data else 0
        logger.info(f"Deleted {deleted_folders_count} folder record from database")
        
        # Delete S3 outputs folder if it exists
        deleted_s3_objects = 0
        if s3_output_folder_url:
            try:
                # Parse S3 output folder URL
                s3_parts = s3_output_folder_url.replace('s3://', '').rstrip('/').split('/', 1)
                bucket = s3_parts[0]
                output_prefix = s3_parts[1] if len(s3_parts) > 1 else ''
                
                # List all objects in the output folder
                s3_client = boto3.client('s3')
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=output_prefix
                )
                
                if 'Contents' in response:
                    # Delete all objects in the output folder
                    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                    
                    if objects_to_delete:
                        # Delete in batches of 1000 (S3 limit)
                        for i in range(0, len(objects_to_delete), 1000):
                            batch = objects_to_delete[i:i+1000]
                            delete_response = s3_client.delete_objects(
                                Bucket=bucket,
                                Delete={'Objects': batch}
                            )
                            deleted_s3_objects += len(batch)
                            logger.info(f"Deleted batch of {len(batch)} objects from S3")
                
                logger.info(f"Deleted {deleted_s3_objects} objects from S3 output folder: {s3_output_folder_url}")
                
            except Exception as s3_error:
                logger.warning(f"Error deleting S3 output folder: {str(s3_error)}")
                # Don't fail the entire operation if S3 deletion fails
        
        # Remove from in-memory tasks if still present
        if task_id in tasks:
            del tasks[task_id]
            logger.info(f"Removed task {task_id} from in-memory tasks")
        
        return {
            "message": "Folder processing data deleted successfully",
            "task_id": task_id,
            "deleted_records": {
                "images": deleted_images_count,
                "folders": deleted_folders_count,
                "s3_objects": deleted_s3_objects
            },
            "preserved": {
                "original_folder": s3_input_folder_url,
                "original_contents": "All original images and files preserved"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting folder processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting folder processing data: {str(e)}")

@app.get("/folders/{task_id}")
async def get_folder_details(task_id: str):
    """Get detailed information about a specific processed folder"""
    try:
        # Get folder info
        folder_result = supabase.table('processed_folders').select('*').eq('task_id', task_id).execute()
        if not folder_result.data:
            raise HTTPException(status_code=404, detail=f"Folder with task_id {task_id} not found")
        
        folder_info = folder_result.data[0]
        
        # Get all images
        images_result = supabase.table('processed_images').select('*').eq('task_id', task_id).order('timestamp').execute()
        images = images_result.data
        total_images = len(images)
        
        # Add signed URLs to images
        images_with_signed_urls = add_signed_urls_to_images(images)
        
        return {
            "folder_info": folder_info,
            "total_images": total_images,
            "images": images_with_signed_urls
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching folder details: {str(e)}")

@app.get("/images/{image_id}")
async def get_image_details(image_id: str):
    """Get detailed information about a specific processed image"""
    try:
        result = supabase.table('processed_images').select('*').eq('id', image_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Image with id {image_id} not found")
        
        image_with_signed_urls = add_signed_urls_to_images([result.data[0]])[0]
        return {"image": image_with_signed_urls}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image details: {str(e)}")

@app.get("/folders/{task_id}/summary")
async def get_folder_summary(task_id: str):
    """Get summary statistics for a processed folder"""
    try:
        # Get folder info
        folder_result = supabase.table('processed_folders').select('*').eq('task_id', task_id).execute()
        if not folder_result.data:
            raise HTTPException(status_code=404, detail=f"Folder with task_id {task_id} not found")
        
        folder_info = folder_result.data[0]
        
        # Get all images for statistics
        images_result = supabase.table('processed_images').select('*').eq('task_id', task_id).execute()
        images = images_result.data
        
        if not images:
            return {
                "folder_info": folder_info,
                "statistics": {
                    "total_images": 0,
                    "images_with_gps": 0,
                    "images_with_detections": 0,
                    "gps_bounds": None,
                    "time_range": None
                }
            }
        
        # Calculate statistics
        images_with_gps = [img for img in images if img.get('latitude') and img.get('longitude')]
        images_with_detections = [img for img in images if img.get('detections')]
        
        # GPS bounds
        gps_bounds = None
        if images_with_gps:
            lats = [float(img['latitude']) for img in images_with_gps]
            lons = [float(img['longitude']) for img in images_with_gps]
            gps_bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons),
                "center": {
                    "lat": sum(lats) / len(lats),
                    "lon": sum(lons) / len(lons)
                }
            }
        
        # Time range
        time_range = None
        timestamps = [img.get('timestamp') for img in images if img.get('timestamp')]
        if timestamps:
            time_range = {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }
        
        return {
            "folder_info": folder_info,
            "statistics": {
                "total_images": len(images),
                "images_with_gps": len(images_with_gps),
                "images_with_detections": len(images_with_detections),
                "gps_bounds": gps_bounds,
                "time_range": time_range
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching folder summary: {str(e)}")

@app.get("/s3-folders")
async def list_s3_folders(bucket: str = "rcsstoragebucket", prefix: str = "fh_sync/bf03aad1-5c1a-464d-8bda-2eac7aeec67f/e1ab4550-4b97-4273-bee4-bccfe1eb87d9/media/"):
    """
    List available folders in S3 and their processing status
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # List objects in the bucket with the given prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
        except ClientError as e:
            raise HTTPException(status_code=403, detail=f"S3 access error: {str(e)}")
        except NoCredentialsError:
            raise HTTPException(status_code=401, detail="S3 credentials not configured")
        
        # Extract folder names from CommonPrefixes
        s3_folders = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                folder_path = prefix_info['Prefix']
                # Extract just the folder name (e.g., "1" from "djiimages/1/")
                folder_name = folder_path.rstrip('/').split('/')[-1]
                s3_url = f"s3://{bucket}/{folder_path}"
                
                s3_folders.append({
                    "folder_name": folder_name,
                    "s3_url": s3_url,
                    "s3_path": folder_path,
                    "total_images": 0,
                    "processed_images": 0
                })
        
        # Check processing status for each folder
        processed_folders_result = supabase.table('processed_folders').select('s3_input_folder_url, folder_name, status, task_id, total_images, processed_images').execute()
        processed_lookup = {folder['s3_input_folder_url']: folder for folder in processed_folders_result.data}
        
        # Combine S3 folder info with processing status
        folders_with_status = []
        for s3_folder in s3_folders:
            s3_url = s3_folder['s3_url']
            processing_info = processed_lookup.get(s3_url)
            
            # Use processed folder name if available, otherwise generate from first image
            if processing_info:
                display_folder_name = processing_info.get('folder_name')
                total_images = processing_info.get('total_images')
                processed_images = processing_info.get('processed_images')
            else:
                # Generate folder name from first image timestamp for unprocessed folders
                display_folder_name = generate_folder_name_from_s3_folder(s3_url)
                # Count images in S3 folder for unprocessed folders
                total_images = count_images_in_s3_folder(s3_url)
                processed_images = 0
            
            folder_info = {
                "folder_name": display_folder_name,
                "s3_url": s3_url,
                "s3_path": s3_folder['s3_path'],
                "is_processed": processing_info is not None,
                "processing_status": processing_info.get('status') if processing_info else None,
                "task_id": processing_info.get('task_id') if processing_info else None,
                "total_images": total_images,
                "processed_images": processed_images
            }
            
            folders_with_status.append(folder_info)
        
        # Sort by timestamp (extracted from folder names)
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
                            from datetime import datetime
                            current_year = datetime.now().year
                            dt_str = f"{current_year} {month_name} {day} {time_part}"
                            dt = datetime.strptime(dt_str, "%Y %B %d %I:%M %p")
                            return dt
                    except Exception as e:
                        logger.warning(f"Error parsing folder name timestamp '{folder_name}': {e}")
                
                # Fallback: return epoch time (will sort to beginning)
                from datetime import datetime
                return datetime(1970, 1, 1)
                
            except Exception as e:
                logger.warning(f"Error extracting timestamp for folder: {e}")
                from datetime import datetime
                return datetime(1970, 1, 1)
        
        # Sort by timestamp (oldest first)
        folders_with_status.sort(key=get_folder_timestamp, reverse=True)
        
        return {
            "bucket": bucket,
            "prefix": prefix,
            "folders": folders_with_status,
            "total_folders": len(folders_with_status),
            "processed_folders": len([f for f in folders_with_status if f['is_processed']])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing S3 folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing S3 folders: {str(e)}")

@app.get("/folders/{task_id}/images/paginated")
async def get_folder_images_paginated(
    task_id: str, 
    page: int = 1, 
    limit: int = 20,
    sort_by: str = "timestamp",
    order: str = "asc"
):
    """Get paginated images for a folder with sorting options"""
    try:
        # Validate sort_by parameter
        valid_sorts = ["timestamp", "filename", "latitude", "longitude"]
        if sort_by not in valid_sorts:
            sort_by = "timestamp"
        
        # Validate order parameter
        order_clause = f"{sort_by}.{order}" if order in ["asc", "desc"] else f"{sort_by}.asc"
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get total count
        count_result = supabase.table('processed_images').select('id', count='exact').eq('task_id', task_id).execute()
        total_images = count_result.count
        
        # Get paginated images
        images_result = supabase.table('processed_images').select('*').eq('task_id', task_id).order(order_clause).range(offset, offset + limit - 1).execute()
        
        total_pages = (total_images + limit - 1) // limit
        
        return {
            "images": images_result.data,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_images": total_images,
                "images_per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "sort": {
                "sort_by": sort_by,
                "order": order
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching paginated images: {str(e)}")

@app.get("/folders/{task_id}/map-data")
async def get_folder_map_data(task_id: str):
    """Get GPS coordinates and basic info for map visualization"""
    try:
        result = supabase.table('processed_images').select(
            'id, filename, latitude, longitude, timestamp, s3_input_url, s3_output_url'
        ).eq('task_id', task_id).execute()
        
        # Filter images with valid GPS coordinates and add signed URLs
        map_points = []
        for img in result.data:
            if img.get('latitude') and img.get('longitude'):
                map_points.append({
                    "id": img['id'],
                    "filename": img['filename'],
                    "lat": float(img['latitude']),
                    "lng": float(img['longitude']),
                    "timestamp": img.get('timestamp'),
                    "s3_input_url": img.get('s3_input_url'),
                    "s3_output_url": img.get('s3_output_url')
                })
        
        # Add signed URLs to map points
        map_points_with_signed_urls = add_signed_urls_to_images(map_points)
        
        # Calculate map bounds
        if map_points:
            lats = [point['lat'] for point in map_points]
            lngs = [point['lng'] for point in map_points]
            bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lngs),
                "west": min(lngs),
                "center": {
                    "lat": sum(lats) / len(lats),
                    "lng": sum(lngs) / len(lngs)
                }
            }
        else:
            bounds = None
        
        return {
            "points": map_points_with_signed_urls,
            "bounds": bounds,
            "total_points": len(map_points_with_signed_urls)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching map data: {str(e)}")

@app.get("/folders/{task_id}/detections-summary")
async def get_folder_detections_summary(task_id: str):
    """Get summary of detections across all images in a folder"""
    try:
        result = supabase.table('processed_images').select('detections, filename').eq('task_id', task_id).execute()
        
        detection_stats = {
            "total_images": len(result.data),
            "images_with_detections": 0,
            "total_detections": 0,
            "detection_types": {},
            "sample_detections": []
        }
        
        for img in result.data:
            detections = img.get('detections', [])
            if detections:
                detection_stats["images_with_detections"] += 1
                
                # Handle both list and single detection formats
                if isinstance(detections, list):
                    for detection in detections:
                        if isinstance(detection, dict):
                            # Count detection types if available
                            detection_type = detection.get('class', detection.get('type', 'unknown'))
                            detection_stats["detection_types"][detection_type] = detection_stats["detection_types"].get(detection_type, 0) + 1
                            detection_stats["total_detections"] += 1
                            
                            # Add to sample detections (limit to first 10)
                            if len(detection_stats["sample_detections"]) < 10:
                                detection_stats["sample_detections"].append({
                                    "filename": img['filename'],
                                    "detection": detection
                                })
        
        return detection_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detection summary: {str(e)}")

@app.get("/folders/{task_id}/timeline")
async def get_folder_timeline(task_id: str):
    """Get chronological timeline of images for a folder"""
    try:
        result = supabase.table('processed_images').select(
            'id, filename, timestamp, latitude, longitude'
        ).eq('task_id', task_id).order('timestamp.asc').execute()
        
        timeline_data = []
        for img in result.data:
            if img.get('timestamp'):
                timeline_data.append({
                    "id": img['id'],
                    "filename": img['filename'],
                    "timestamp": img['timestamp'],
                    "has_gps": bool(img.get('latitude') and img.get('longitude')),
                    "gps": {
                        "lat": float(img['latitude']) if img.get('latitude') else None,
                        "lng": float(img['longitude']) if img.get('longitude') else None
                    } if img.get('latitude') and img.get('longitude') else None
                })
        
        # Calculate time spans
        time_info = None
        if timeline_data:
            timestamps = [item['timestamp'] for item in timeline_data]
            time_info = {
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration_minutes": None  # Could calculate if needed
            }
        
        return {
            "timeline": timeline_data,
            "time_info": time_info,
            "total_images": len(timeline_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching timeline: {str(e)}")

@app.get("/folders/{task_id}/export")
async def export_folder_data(task_id: str, format: str = "json"):
    """Export folder data in various formats"""
    try:
        # Get folder info
        folder_result = supabase.table('processed_folders').select('*').eq('task_id', task_id).execute()
        if not folder_result.data:
            raise HTTPException(status_code=404, detail=f"Folder with task_id {task_id} not found")
        
        # Get all images
        images_result = supabase.table('processed_images').select('*').eq('task_id', task_id).order('timestamp').execute()
        
        export_data = {
            "folder_info": folder_result.data[0],
            "images": images_result.data,
            "export_metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_images": len(images_result.data),
                "format": format
            }
        }
        
        if format.lower() == "csv":
            # For CSV, we'd need to flatten the data structure
            # This would require additional processing
            return {"message": "CSV export not yet implemented", "data": export_data}
        else:
            return export_data
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

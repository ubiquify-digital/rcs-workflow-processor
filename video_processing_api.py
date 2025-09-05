# Set ALL environment variables BEFORE any imports
import os
os.environ["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"] = "True"
# OpenAI API key - loaded from environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Use .env file instead

# Configuration from environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "MQ1Wd6PJGMPBMCvxsCS6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "rcs-k9i1w")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "awais-detect-trash")

# Now import everything else
import cv2
import json

import tempfile
import shutil
import numpy as np
import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,  field_validator
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from inference import InferencePipeline
import uuid
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# S3 Configuration - API controls output bucket for security
DEFAULT_OUTPUT_BUCKET = os.getenv("DEFAULT_OUTPUT_BUCKET", "processed-videos-bucket")
ALLOW_CUSTOM_OUTPUT_BUCKET = os.getenv("ALLOW_CUSTOM_OUTPUT_BUCKET", "false").lower() == "true"


app = FastAPI(title="Video Processing API", description="API for processing videos with object detection")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

default_fps = 1

# Request/Response models
class VideoProcessRequest(BaseModel):
    s3_url: str  # Can be s3:// URL or signed HTTPS URL
    output_s3_bucket: Optional[str] = None  # Optional - uses default if not provided
    output_s3_key: Optional[str] = None  # If not provided, will auto-generate
    generate_signed_output_url: Optional[bool] = True  # Return signed URL for output
    signed_url_expiry_hours: Optional[int] = 24  # Expiry for output signed URL
    max_fps: Optional[int] = default_fps
    output_fps: Optional[int] = default_fps # should be same as max_fps

    @field_validator('s3_url')
    @classmethod
    def validate_s3_url(cls, v):
        if not (v.startswith('s3://') or v.startswith('https://')):
            raise ValueError('s3_url must start with s3:// or https://')
        return v

class VideoProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[str] = None
    s3_output_url: Optional[str] = None
    signed_output_url: Optional[str] = None  # Signed URL if requested
    s3_detections_url: Optional[str] = None  # Unified detections JSON
    signed_detections_url: Optional[str] = None  # Signed URL for detections
    error: Optional[str] = None

# In-memory task storage (in production, use Redis or database)
tasks = {}

class VideoProcessor:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.frame_counter = 0
        self.temp_dir = None
        self.output_folder = None
        
    def make_serializable(self, obj):
        """
        Convert custom objects (Detections, WorkflowImageData, datetime, numpy arrays) into JSON-friendly structures.
        Excludes large image data to keep JSON files manageable.
        """
        if hasattr(obj, "__dict__"):
            # Filter out image-related attributes to avoid saving large data
            filtered_dict = {}
            for k, v in obj.__dict__.items():
                # Skip image data attributes
                if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image']:
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
                if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image']:
                    filtered_dict[k] = f"<excluded_image_data_{type(v).__name__}>"
                else:
                    filtered_dict[k] = self.make_serializable(v)
            return filtered_dict
        elif isinstance(obj, np.ndarray):
            # For numpy arrays, check if it's likely image data (large arrays)
            if obj.size > 1000:  # Skip large arrays (likely images)
                return f"<excluded_large_array_shape_{obj.shape}>"
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)  # Convert numpy integers to Python int
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)  # Convert numpy floats to Python float
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()  # convert datetime to ISO string
        else:
            return obj

    def video_sink(self, result, video_frame):
        """Callback function for processing each frame"""
        try:
            # Save the frame
            if result.get("output_image") or result.get("label_visualization") or result.get("car_model_predictions") or result.get("license_plate_model_predictions"): 
                frame_path = os.path.join(self.output_folder, "frames", f"frame_{self.frame_counter:05d}.png")
                if result.get("output_image"):
                    cv2.imwrite(frame_path, result["output_image"].numpy_image)
                elif result.get("car_model_predictions"):
                    cv2.imwrite(frame_path, result["car_model_predictions"].numpy_image)    
                elif result.get("license_plate_model_predictions"):
                    cv2.imwrite(frame_path, result["license_plate_model_predictions"].numpy_image)
                elif result.get("label_visualization"):
                    cv2.imwrite(frame_path, result["label_visualization"].numpy_image)
                else:
                    cv2.imwrite(frame_path, result["fallback_visual"].numpy_image)

            # Serialize the result
            serializable_result = self.make_serializable(result)

            # Save JSON output
            json_path = os.path.join(self.output_folder, "json_outputs", f"frame_{self.frame_counter:05d}.json")
            with open(json_path, "w") as f:
                json.dump(serializable_result, f, indent=2)

            self.frame_counter += 1
            logger.info(f"Processed frame {self.frame_counter} for task {self.task_id}")
            
            # Update task progress
            tasks[self.task_id]["progress"] = f"Processed {self.frame_counter} frames"
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_counter}: {str(e)}")
            raise

    def download_from_s3(self, s3_url: str) -> str:
        """Download video from S3 URL or signed URL"""
        try:
            local_path = os.path.join(self.temp_dir, 'input_video.mp4')
            
            # Check if it's a signed URL (HTTPS)
            if s3_url.startswith('https://'):
                logger.info(f"Downloading from signed URL: {s3_url[:50]}...")
                # Use requests to download from signed URL
                import requests
                response = requests.get(s3_url, stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            elif s3_url.startswith('s3://'):
                # Parse S3 URL
                s3_parts = s3_url.replace('s3://', '').split('/', 1)
                bucket_name = s3_parts[0]
                object_key = s3_parts[1]
                
                # Download file using boto3
                s3_client = boto3.client('s3')
                logger.info(f"Downloading from S3: {bucket_name}/{object_key}")
                s3_client.download_file(bucket_name, object_key, local_path)
                
            else:
                # Try to parse as HTTPS S3 URL
                if 's3.amazonaws.com' in s3_url or 's3.' in s3_url:
                    parts = s3_url.replace('https://', '').split('/')
                    bucket = parts[0].split('.')[0]
                    key = '/'.join(parts[1:])
                    
                    s3_client = boto3.client('s3')
                    logger.info(f"Downloading from S3: {bucket}/{key}")
                    s3_client.download_file(bucket, key, local_path)
                else:
                    raise ValueError("Invalid URL format. Use s3:// or https:// signed URL")
            
            return local_path
            
        except (NoCredentialsError, ClientError) as e:
            raise HTTPException(status_code=400, detail=f"S3 access error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download error: {str(e)}")

    def create_video_from_frames(self, fps: int) -> str:
        """Create video from processed frames"""
        try:
            frames_dir = os.path.join(self.output_folder, "frames")
            output_video_path = os.path.join(self.temp_dir, "processed_video.mp4")
            
            # Get sorted list of frame files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            
            if not frame_files:
                raise ValueError("No frames found to create video")
            
            # Read the first frame to get video dimensions
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            height, width, layers = first_frame.shape
            
            # Define the video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Write all frames
            for frame_file in tqdm(frame_files, desc=f"Creating video for task {self.task_id}"):
                frame_path = os.path.join(frames_dir, frame_file)
                img = cv2.imread(frame_path)
                video_writer.write(img)
            
            video_writer.release()
            logger.info(f"Video created: {output_video_path}")
            
            return output_video_path
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise

    def upload_to_s3(self, local_video_path: str, bucket: str, key: str) -> str:
        """Upload processed video to S3 and return S3 URL"""
        try:
            s3_client = boto3.client('s3')
            
            logger.info(f"Uploading to S3: {bucket}/{key}")
            s3_client.upload_file(local_video_path, bucket, key)
            
            # Return S3 URL
            s3_url = f"s3://{bucket}/{key}"
            logger.info(f"Successfully uploaded to: {s3_url}")
            
            return s3_url
            
        except (NoCredentialsError, ClientError) as e:
            raise HTTPException(status_code=400, detail=f"S3 upload error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    def generate_signed_url(self, bucket: str, key: str, expiry_hours: int = 24) -> str:
        """Generate a signed URL for downloading from S3"""
        try:
            s3_client = boto3.client('s3')
            
            signed_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiry_hours * 3600  # Convert hours to seconds
            )
            
            logger.info(f"Generated signed URL for {bucket}/{key} (expires in {expiry_hours}h)")
            return signed_url
            
        except Exception as e:
            logger.error(f"Error generating signed URL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Signed URL generation error: {str(e)}")

    def create_unified_detections_json(self) -> str:
        """Create a simplified unified JSON file with raw frame outputs"""
        try:
            json_outputs_dir = os.path.join(self.output_folder, "json_outputs")
            unified_detections = {
                "video_metadata": {
                    "total_frames": self.frame_counter,
                    "processing_date": datetime.datetime.now().isoformat(),
                    "task_id": self.task_id
                },
                "detections_by_frame": []
            }
            
            # Process each frame's JSON
            json_files = sorted([f for f in os.listdir(json_outputs_dir) if f.endswith('.json')])
            
            for i, json_file in enumerate(json_files):
                frame_path = os.path.join(json_outputs_dir, json_file)
                with open(frame_path, 'r') as f:
                    frame_data = json.load(f)
                
                # Simple frame entry with raw JSON output
                frame_entry = {
                    "frame_number": i + 1,  # Sequential frame numbers for processed video
                    "output": frame_data  # Raw JSON output from workflow
                }
                
                unified_detections['detections_by_frame'].append(frame_entry)
            
            # Save unified JSON
            unified_json_path = os.path.join(self.temp_dir, "unified_detections.json")
            with open(unified_json_path, 'w') as f:
                json.dump(unified_detections, f, indent=2)
            
            logger.info(f"Created simplified unified detections JSON: {unified_json_path}")
            return unified_json_path
            
        except Exception as e:
            logger.error(f"Error creating unified detections JSON: {str(e)}")
            raise

    def process_video(self, s3_url: str, output_s3_bucket: str, output_s3_key: str, generate_signed_url: bool, signed_url_expiry: int, max_fps: int, output_fps: int):
        """Main processing function"""
        try:
            # Update task status
            tasks[self.task_id]["status"] = "processing"
            tasks[self.task_id]["progress"] = "Initializing..."
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix=f"video_process_{self.task_id}_")
            self.output_folder = os.path.join(self.temp_dir, "output_frames")
            
            # Create output directories
            os.makedirs(os.path.join(self.output_folder, "frames"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder, "json_outputs"), exist_ok=True)
            
            # Download video from S3
            tasks[self.task_id]["progress"] = "Downloading video from S3..."
            video_path = self.download_from_s3(s3_url)
            
            # Process video with inference pipeline
            tasks[self.task_id]["progress"] = "Processing video frames..."
            pipeline = InferencePipeline.init_with_workflow(
                api_key=ROBOFLOW_API_KEY,
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                video_reference=video_path,
                max_fps=max_fps,
                on_prediction=self.video_sink,
            )
            
            pipeline.start()
            pipeline.join()
            
            # Create output video
            tasks[self.task_id]["progress"] = "Creating output video..."
            output_video_path = self.create_video_from_frames(output_fps)
            
            # Create unified detections JSON
            tasks[self.task_id]["progress"] = "Creating unified detections JSON..."
            unified_json_path = self.create_unified_detections_json()
            
            # Upload video to S3
            tasks[self.task_id]["progress"] = "Uploading video to S3..."
            s3_output_url = self.upload_to_s3(output_video_path, output_s3_bucket, output_s3_key)
            
            # Upload detections JSON to S3
            tasks[self.task_id]["progress"] = "Uploading detections JSON to S3..."
            # Place detections.json in same folder as video: processed/{task_id}/detections.json
            if '/video.mp4' in output_s3_key:
                detections_s3_key = output_s3_key.replace('/video.mp4', '/detections.json')
            else:
                # Fallback for custom keys
                detections_s3_key = output_s3_key.replace('.mp4', '_detections.json')
            s3_detections_url = self.upload_to_s3(unified_json_path, output_s3_bucket, detections_s3_key)
            
            # Generate signed URLs if requested
            signed_output_url = None
            signed_detections_url = None
            if generate_signed_url:
                tasks[self.task_id]["progress"] = "Generating signed URLs..."
                signed_output_url = self.generate_signed_url(output_s3_bucket, output_s3_key, signed_url_expiry)
                signed_detections_url = self.generate_signed_url(output_s3_bucket, detections_s3_key, signed_url_expiry)
            
            # Update task status
            tasks[self.task_id]["status"] = "completed"
            tasks[self.task_id]["progress"] = f"Completed! Processed {self.frame_counter} frames"
            tasks[self.task_id]["s3_output_url"] = s3_output_url
            tasks[self.task_id]["signed_output_url"] = signed_output_url
            tasks[self.task_id]["s3_detections_url"] = s3_detections_url
            tasks[self.task_id]["signed_detections_url"] = signed_detections_url
            
            logger.info(f"Task {self.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing video for task {self.task_id}: {str(e)}")
            tasks[self.task_id]["status"] = "failed"
            tasks[self.task_id]["error"] = str(e)
        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video_endpoint(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a video from S3 URL and return processed video
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task
        tasks[task_id] = {
            "status": "queued",
            "progress": None,
            "s3_output_url": None,
            "signed_output_url": None,
            "s3_detections_url": None,
            "signed_detections_url": None,
            "error": None,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Create processor and start background task
        processor = VideoProcessor(task_id)
        
        # Determine output bucket - use default if not provided or not allowed
        if request.output_s3_bucket and ALLOW_CUSTOM_OUTPUT_BUCKET:
            output_s3_bucket = request.output_s3_bucket
            logger.info(f"Using custom output bucket: {output_s3_bucket}")
        else:
            output_s3_bucket = DEFAULT_OUTPUT_BUCKET
            if request.output_s3_bucket and not ALLOW_CUSTOM_OUTPUT_BUCKET:
                logger.warning(f"Custom bucket {request.output_s3_bucket} requested but not allowed. Using default: {output_s3_bucket}")
        
        # Generate output S3 key if not provided - organized by task ID
        if request.output_s3_key:
            output_s3_key = request.output_s3_key
        else:
            # Create organized folder structure: processed/{task_id}/video.mp4
            output_s3_key = f"processed/{task_id}/video.mp4"
        
        # Submit to thread pool
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor, 
            processor.process_video, 
            str(request.s3_url),
            output_s3_bucket,
            output_s3_key,
            request.generate_signed_output_url,
            request.signed_url_expiry_hours,
            request.max_fps, 
            request.output_fps
        )
        
        return VideoProcessResponse(
            task_id=task_id,
            status="queued",
            message="Video processing started. Use /status/{task_id} to check progress."
        )
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status of a video processing task
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data["progress"],
        s3_output_url=task_data["s3_output_url"],
        signed_output_url=task_data["signed_output_url"],
        s3_detections_url=task_data["s3_detections_url"],
        signed_detections_url=task_data["signed_detections_url"],
        error=task_data["error"]
    )


@app.get("/")
async def root():
    """
    API health check
    """
    return {"message": "Video Processing API is running", "version": "1.0.0"}

@app.get("/config")
async def get_config():
    """
    Get API configuration - useful for frontend to know bucket policies
    """
    return {
        "default_output_bucket": DEFAULT_OUTPUT_BUCKET,
        "allow_custom_output_bucket": ALLOW_CUSTOM_OUTPUT_BUCKET,
        "supported_input_formats": ["s3://", "https://"],
        "max_signed_url_expiry_hours": 168,  # 1 week
        "default_signed_url_expiry_hours": 24
    }

class SignedUrlRequest(BaseModel):
    bucket: str
    key: str
    expiry_hours: Optional[int] = 24
    operation: Optional[str] = "get_object"  # get_object or put_object

class SignedUrlResponse(BaseModel):
    signed_url: str
    expires_at: str

@app.post("/generate-signed-url", response_model=SignedUrlResponse)
async def generate_signed_url_endpoint(request: SignedUrlRequest):
    """
    Generate a signed URL for S3 operations
    Useful for frontend to get upload/download URLs
    """
    try:
        s3_client = boto3.client('s3')
        
        # Generate signed URL
        signed_url = s3_client.generate_presigned_url(
            request.operation,
            Params={'Bucket': request.bucket, 'Key': request.key},
            ExpiresIn=request.expiry_hours * 3600
        )
        
        # Calculate expiration time
        expires_at = (datetime.datetime.now() + datetime.timedelta(hours=request.expiry_hours)).isoformat()
        
        return SignedUrlResponse(
            signed_url=signed_url,
            expires_at=expires_at
        )
        
    except (NoCredentialsError, ClientError) as e:
        raise HTTPException(status_code=400, detail=f"S3 access error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signed URL: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

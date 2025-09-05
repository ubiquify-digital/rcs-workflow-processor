# Video Processing API

A FastAPI-based service that processes videos using object detection and returns processed videos.

## Features

- Accept S3 URLs as input for video files
- Process videos using inference pipeline with object detection
- Generate processed video with visualizations
- Create unified JSON with all detections and metadata
- Asynchronous processing with task tracking
- Upload processed videos and detection data back to S3
- Return S3 URLs for both video and detections
- No local file storage - fully cloud-based workflow

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up AWS credentials (for S3 access):

The API uses boto3 which supports multiple credential methods. Choose one:

**Option 1: AWS CLI Configuration (Recommended)**
```bash
aws configure
# Enter your AWS Access Key ID, Secret Key, and default region
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
export AWS_DEFAULT_REGION=us-east-1
```

**Option 3: IAM Role (for EC2 instances)**
- Attach an IAM role to your EC2 instance with S3 permissions
- No additional configuration needed

**Option 4: AWS Credentials File**
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key_here
aws_secret_access_key = your_secret_access_key_here
```

And `~/.aws/config`:
```ini
[default]
region = us-east-1
```

**Required S3 Permissions:**
Your AWS credentials need the following permissions:
- `s3:GetObject` on input buckets
- `s3:PutObject` on output buckets
- `s3:ListBucket` (optional, for better error messages)

## Usage

### Start the API server:

**Method 1: Direct execution**
```bash
python video_processing_api.py
```

**Method 2: With environment variables**
```bash
# Copy and edit the example environment file
cp env.example .env
# Edit .env with your actual credentials

# Load environment variables and start
export $(cat .env | xargs) && python video_processing_api.py
```

**Method 3: Set variables inline**
```bash
AWS_ACCESS_KEY_ID=your_key AWS_SECRET_ACCESS_KEY=your_secret python video_processing_api.py
```

The API will be available at `http://localhost:8000` (or the port specified in API_PORT)

### API Endpoints:

#### 1. Process Video
**POST** `/process-video`

**Minimal Request (Recommended):**
```json
{
  "s3_url": "s3://your-input-bucket/path/to/video.mp4"
}
```

**Supported URL formats:**
- `s3://bucket-name/path/to/video.mp4` (S3 URI - uses your AWS credentials)
- `https://bucket-name.s3.region.amazonaws.com/path/to/video.mp4` (HTTPS S3 URL - including signed URLs)

**Full Request (all optional parameters):**
```json
{
  "s3_url": "s3://your-input-bucket/path/to/video.mp4",
  "output_s3_bucket": "custom-output-bucket",
  "output_s3_key": "processed_videos/output.mp4",
  "generate_signed_output_url": true,
  "signed_url_expiry_hours": 24,
  "max_fps": 1,
  "output_fps": 1
}
```

**Note**: All parameters except `s3_url` are optional. The API will use sensible defaults.

**Parameters:**
- `s3_url`: Can be either `s3://` URL or signed HTTPS URL
- `output_s3_bucket`: **Optional** - uses API default if not provided
- `output_s3_key`: Optional - auto-generates if not provided
- `generate_signed_output_url`: Set to `true` to get a signed download URL
- `signed_url_expiry_hours`: Expiry time for signed URLs (default: 24 hours)

> **Note**: Custom output buckets are only allowed if `ALLOW_CUSTOM_OUTPUT_BUCKET=true`

Response:
```json
{
  "task_id": "uuid-task-id",
  "status": "queued",
  "message": "Video processing started. Use /status/{task_id} to check progress."
}
```

#### 2. Check Task Status
**GET** `/status/{task_id}`

Response:
```json
{
  "task_id": "uuid-task-id",
  "status": "completed",
  "progress": "Completed! Processed 150 frames",
  "s3_output_url": "s3://your-output-bucket/processed_videos/output.mp4",
  "signed_output_url": "https://your-output-bucket.s3.amazonaws.com/processed_videos/output.mp4?...",
  "s3_detections_url": "s3://your-output-bucket/processed_videos/output_detections.json",
  "signed_detections_url": "https://your-output-bucket.s3.amazonaws.com/processed_videos/output_detections.json?...",
  "error": null
}
```

#### 3. Generate Signed URL
**POST** `/generate-signed-url`

Useful for frontend to get upload/download URLs without exposing AWS credentials.

Request body:
```json
{
  "bucket": "your-bucket",
  "key": "path/to/file.mp4",
  "expiry_hours": 24,
  "operation": "get_object"
}
```

Response:
```json
{
  "signed_url": "https://your-bucket.s3.amazonaws.com/path/to/file.mp4?AWSAccessKeyId=...",
  "expires_at": "2024-01-02T12:00:00"
}
```

#### 4. Get Configuration
**GET** `/config`

Get API configuration to understand bucket policies and defaults.

Response:
```json
{
  "default_output_bucket": "processed-videos-bucket",
  "allow_custom_output_bucket": false,
  "supported_input_formats": ["s3://", "https://"],
  "max_signed_url_expiry_hours": 168,
  "default_signed_url_expiry_hours": 24
}
```

#### 5. Health Check
**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Detection JSON Output Format

The API returns detection data in a simplified, clean format that preserves all raw workflow output:

```json
{
  "video_metadata": {
    "total_frames": 102,
    "processing_date": "2025-09-05T13:57:06.123456",
    "task_id": "683008e4-1ea2-4ac8-89a9-08db9b8325c0"
  },
  "detections_by_frame": [
    {
      "frame_number": 1,
      "output": {
        "license_plate_model_predictions": [...],
        "car_model_predictions": {...},
        "label_visualization": {...},
        "license_plates": [...],
        "license_plate_text": [...],
        "cars": [...],
        // ... complete raw workflow output preserved
      }
    },
    {
      "frame_number": 2,
      "output": {
        // ... raw workflow output for frame 2
      }
    }
    // ... for all processed frames
  ]
}
```

### Benefits of Simplified Format:

- **🎯 Clean Structure**: Just `frame_number` + `output` - easy to iterate
- **🔧 Complete Data**: All raw workflow output preserved without transformation
- **⚡ Future-Proof**: Works with any Roboflow workflow format
- **🚀 Frontend-Friendly**: Simple parsing, extract only what you need
- **📊 Flexible**: Access any workflow-specific data (detections, OCR, classifications)

This format allows frontends to easily process frame-by-frame data while maintaining access to all original workflow outputs.

## Signed URL Workflow (Recommended for Frontend)

For production applications, use signed URLs to avoid exposing AWS credentials:

```bash
# 1. Frontend gets signed URL for input video upload
curl -X POST "http://localhost:8000/generate-signed-url" \
  -H "Content-Type: application/json" \
  -d '{
    "bucket": "input-bucket",
    "key": "videos/input.mp4",
    "operation": "put_object",
    "expiry_hours": 1
  }'

# 2. Frontend uploads video to signed URL (not shown)

# 3. Frontend starts processing with signed input URL (minimal request)
curl -X POST "http://localhost:8000/process-video" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "https://input-bucket.s3.amazonaws.com/videos/input.mp4?AWSAccessKeyId=...",
    "generate_signed_output_url": true
  }'

# 4. Check status - will include signed_output_url when completed
curl "http://localhost:8000/status/task-id"
```

## Example Usage with curl:

```bash
# 1. Start processing (minimal - uses default output bucket)
curl -X POST "http://localhost:8000/process-video" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_url": "s3://your-input-bucket/input-video.mp4"
  }'

# 2. Check status (replace task_id with actual ID from step 1)
curl "http://localhost:8000/status/your-task-id"

# The response will include s3_output_url when completed
# You can then download directly from S3 or use the S3 URL as needed
```

## Configuration

The API uses the following configuration from the original scripts:
- Roboflow API key: `MQ1Wd6PJGMPBMCvxsCS6`
- Workspace: `rcs-k9i1w`
- Workflow ID: `awais-detect-trash`

Update these values in `video_processing_api.py` as needed for your setup.

## Project Structure

```
/home/ubuntu/workflows/
├── video_processing_api.py          # Main FastAPI application
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
├── .env                           # Environment variables (create from .env.example)
├── env.example                    # Example environment variables template
├── iam-policy.json               # Example IAM policy for S3 permissions
├── README.md                     # This documentation
├── video-processing-api.service  # Systemd service configuration
├── archived_scripts/             # Legacy scripts (archived)
│   ├── excute.py                 # Original frame processing script
│   ├── joinerUpdate.py          # Original video creation script  
│   └── test.py                  # Test utilities
└── venv/                        # Python virtual environment
    ├── bin/                     # Executables and activation scripts
    ├── lib/                     # Installed packages
    └── ...
```

### Key Files:

- **`video_processing_api.py`** - Main API server with all endpoints and processing logic
- **`requirements.txt`** - All Python dependencies for the project
- **`.env`** - Your actual environment variables (create from `env.example`)
- **`env.example`** - Template showing all required environment variables
- **`video-processing-api.service`** - Systemd service file for production deployment
- **`archived_scripts/`** - Contains the original scripts that were combined into the API
- **`venv/`** - Python virtual environment with all dependencies installed

## Production Deployment (Systemd Service)

For production deployment, the API can be run as a systemd service for automatic startup and management.

### Setup Service

1. **Copy service file:**
   ```bash
   sudo cp video-processing-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable video-processing-api
   ```

2. **Start the service:**
   ```bash
   sudo systemctl start video-processing-api
   ```

### Service Management Commands

```bash
# Check service status
sudo systemctl status video-processing-api

# View real-time logs
sudo journalctl -u video-processing-api -f

# Restart service (after code changes)
sudo systemctl restart video-processing-api

# Stop service
sudo systemctl stop video-processing-api

# Disable auto-start
sudo systemctl disable video-processing-api
```

### Making Code Changes

When you modify the code:

1. **Edit your Python files** (the service runs from the virtual environment)
2. **Restart the service** to apply changes:
   ```bash
   sudo systemctl restart video-processing-api
   ```
3. **Check logs** to verify restart:
   ```bash
   sudo journalctl -u video-processing-api -f
   ```

**Note**: The service automatically uses the virtual environment at `/home/ubuntu/workflows/venv/` as specified in the service file.

### Public Access

Once running as a service, the API is accessible at:
- **Local**: http://localhost:8000
- **Public**: http://YOUR_SERVER_IP:8000 (ensure port 8000 is open in security groups)

## Notes

- Videos are processed asynchronously in the background
- Temporary files are automatically cleaned up after processing
- Large image data is excluded from JSON outputs to keep files manageable
- The API supports both S3:// URLs and HTTPS S3 URLs
- Processed videos are uploaded to S3 and original local copies are cleaned up
- Ensure your AWS credentials have read access to input bucket and write access to output bucket
- By default, API controls the output bucket for security (configure via `DEFAULT_OUTPUT_BUCKET`)
- Frontend can specify custom buckets only if `ALLOW_CUSTOM_OUTPUT_BUCKET=true`
- Service runs in virtual environment automatically - no need to activate it manually
- CORS is enabled for all origins to support frontend integration


# QR Processing API

A FastAPI-based service for detecting and decoding QR codes from drone imagery stored in AWS S3. This API processes entire folders of images, extracts GPS coordinates from EXIF data, and maps QR code locations with high precision.

## 🎯 Overview

The QR Processing API is designed to process aerial drone imagery containing QR codes. It automatically:
- Downloads images from S3 folders
- Extracts GPS coordinates from image EXIF data
- Detects QR codes using Roboflow's computer vision workflow
- Decodes QR codes using multiple image processing techniques
- Returns comprehensive results with location mapping

## 🚀 Features

- **Folder-based Processing**: Process entire S3 folders containing multiple images
- **GPS Coordinate Extraction**: Automatic extraction of latitude/longitude from drone imagery
- **Robust QR Detection**: Uses Roboflow's AI-powered QR detection workflow
- **Multi-method Decoding**: 6 different image processing methods for reliable QR code decoding
- **Local Inference Server Support**: Prioritizes local inference server for better performance
- **Signed Image URLs**: Returns pre-signed S3 URLs for direct image access (automatically refreshed for cached results)
- **Intelligent Caching**: Database-backed result caching for instant repeated requests
- **Cache Management**: Full control over cached results with management endpoints
- **Error Handling**: Continues processing even if individual images fail
- **Comprehensive Results**: Detailed response with per-image results and confidence scores

## 🌐 Live Server

The QR Processing API is currently deployed and accessible at:
- **Server IP**: `13.48.174.160:8002`
- **Health Check**: http://13.48.174.160:8002/health
- **S3 Folders**: http://13.48.174.160:8002/s3-folders

## 📋 Requirements

### System Dependencies
```bash
# Install system dependency for QR code decoding
sudo apt-get update
sudo apt-get install -y libzbar0
```

### Python Dependencies
```bash
pip install -r requirements_qr.txt
```

### Environment Variables
```bash
# Roboflow Configuration
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_WORKSPACE=your_workspace_name
QR_WORKFLOW_ID=qr-workflow

# AWS Configuration (handled automatically via environment or IAM role)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=eu-north-1
```

## 🔧 Installation

1. **Clone and setup**:
```bash
cd /home/ubuntu/workflows/qr_code_app
```

2. **Install dependencies**:
```bash
pip install -r requirements_qr.txt
sudo apt-get install -y libzbar0
```

3. **Configure environment**:
```bash
# Set up your .env file with Roboflow credentials
echo "ROBOFLOW_API_KEY=your_api_key_here" > .env
echo "ROBOFLOW_WORKSPACE=your_workspace" >> .env
echo "QR_WORKFLOW_ID=qr-workflow" >> .env
```

4. **Start the API**:
```bash
python qr_processing_api.py
```

The API will start on `http://0.0.0.0:8002`

## 📖 API Documentation

### Base URL
```
http://localhost:8002
```

### Endpoints

#### Process QR Codes from S3 Folder
```http
POST /process-qr
```

**Request Body:**
```json
{
  "s3_folder_url": "s3://bucket-name/path/to/folder/",
  "rerun": false  // Optional: Set to true to force reprocessing (ignore cache)
}
```

**Response:**
```json
{
  "success": true,
  "message": "Processed 17/17 images, found 12 QR codes",
  "total_images": 17,
  "processed_images": 17,
  "results": [
    {
      "image_name": "DJI_20250915113838_0007_V.jpeg",
      "image_url": "https://rcsstoragebucket.s3.amazonaws.com/path/to/image.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&...",
      "image_coordinates": {
        "latitude": 24.949416194444446,
        "longitude": 55.51372725
      },
      "qr_results": [
        {
          "content": "https://qrto.org/9CmNYn",
          "confidence": 0.9004175662994385
        }
      ],
      "success": true,
      "error": null
    }
  ],
  "error": null
}
```

#### List S3 Folders
```http
GET /s3-folders
```

**Query Parameters:**
- `bucket` (optional): S3 bucket name (default: "rcsstoragebucket")
- `prefix` (optional): S3 path prefix (default: "fh_sync/bf03aad1-5c1a-464d-8bda-2eac7aeec67f/e1ab4550-4b97-4273-bee4-bccfe1eb87d9/media/")

**Response:**
```json
{
  "bucket": "rcsstoragebucket",
  "prefix": "fh_sync/.../media/",
  "folders": [
    {
      "folder_name": "Run September 15 11:38 AM",
      "s3_url": "s3://bucket/path/to/folder/",
      "s3_path": "path/to/folder/",
      "total_images": 17,
      "has_images": true
    }
  ],
  "total_folders": 37,
  "folders_with_images": 37
}
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

## 🎯 Usage Examples

### List Available Folders
```bash
# List all available folders (local)
curl -X GET "http://localhost:8002/s3-folders"

# List folders (remote server)
curl -X GET "http://13.48.174.160:8002/s3-folders"

# List folders with custom bucket/prefix
curl -X GET "http://13.48.174.160:8002/s3-folders?bucket=mybucket&prefix=mypath/"
```

### Process QR Codes
```bash
# Normal processing (uses cache if available)
curl -X POST "http://13.48.174.160:8002/process-qr" \
  -H "Content-Type: application/json" \
  -d '{"s3_folder_url": "s3://rcsstoragebucket/fh_sync/.../media/folder-id/"}'

# Force reprocessing (ignore cache)
curl -X POST "http://13.48.174.160:8002/process-qr" \
  -H "Content-Type: application/json" \
  -d '{"s3_folder_url": "s3://rcsstoragebucket/fh_sync/.../media/folder-id/", "rerun": true}'
```

### Python Example
```python
import requests
import json

# 1. List available folders
folders_response = requests.get("http://13.48.174.160:8002/s3-folders")
folders_data = folders_response.json()

print(f"Available folders: {folders_data['total_folders']}")
for folder in folders_data['folders'][:5]:  # Show first 5
    print(f"- {folder['folder_name']} ({folder['total_images']} images)")

# 2. Select and process a folder
selected_folder = folders_data['folders'][0]['s3_url']  # Use first folder
process_response = requests.post(
    "http://13.48.174.160:8002/process-qr",
    json={
        "s3_folder_url": selected_folder,
        "rerun": False  # Set to True to force reprocessing
    }
)
result = process_response.json()

print(f"\nProcessed: {result['processed_images']}/{result['total_images']} images")
print(f"Found: {sum(len(r['qr_results']) for r in result['results'])} QR codes")

# Print QR codes with locations and image URLs
for image_result in result['results']:
    if image_result['qr_results']:
        coords = image_result['image_coordinates']
        print(f"\n{image_result['image_name']}:")
        print(f"  Image URL: {image_result['image_url']}")
        print(f"  Location: {coords['latitude']:.6f}, {coords['longitude']:.6f}")
        for qr in image_result['qr_results']:
            print(f"  QR Code: {qr['content']} (confidence: {qr['confidence']:.2f})")

# Example: Download an image with QR codes
import requests
from PIL import Image
import io

first_result_with_qr = next((r for r in result['results'] if r['qr_results']), None)
if first_result_with_qr:
    image_response = requests.get(first_result_with_qr['image_url'])
    image = Image.open(io.BytesIO(image_response.content))
    # Now you can display or process the image
    print(f"Downloaded image: {image.size}")
```

## 🔍 Technical Details

### Image Processing Pipeline
1. **S3 Folder Listing**: Lists all image files (.jpg, .jpeg, .png, .tiff, .tif) in the specified folder
2. **Image Download**: Downloads each image locally for processing
3. **GPS Extraction**: Extracts GPS coordinates from EXIF data using PIL
4. **QR Detection**: Uses Roboflow's computer vision workflow to detect QR code regions
5. **QR Decoding**: Applies 6 different image processing methods for robust decoding:
   - Direct grayscale conversion
   - Histogram equalization
   - Gaussian blur + contrast enhancement
   - Morphological operations (erosion/dilation)
   - OTSU thresholding
   - Adaptive thresholding with scaling
6. **Cleanup**: Removes temporary files after processing

### Inference Server Priority
The API automatically detects and prioritizes local inference servers:
- **Local Server**: `http://localhost:9001` (preferred for better performance)
- **Cloud API**: `https://detect.roboflow.com` (fallback)

### S3 Folder Discovery
- **Smart Folder Naming**: Extracts timestamp from first image in each folder and generates human-readable names (e.g., "Run September 15 11:38 AM")
- **Chronological Sorting**: Folders are sorted by capture time, newest first
- **Image Counting**: Automatically counts images in each folder
- **Fallback Handling**: Uses original folder names if timestamp extraction fails

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)

### GPS Coordinate System
- **Format**: Decimal degrees
- **Precision**: Up to 6 decimal places
- **Source**: EXIF GPS data from drone imagery

### Signed Image URLs
- **Purpose**: Direct access to processed images for display or download
- **Expiration**: 1 hour (3600 seconds)
- **Format**: Pre-signed S3 URLs with AWS authentication
- **Fallback**: Returns original S3 URL if signing fails

## 🚀 Production Deployment (Systemd Service)

The QR Processing API runs as a systemd service for production deployment with automatic startup and management.

### **Service Management**
```bash
# Check service status
sudo systemctl status qr-processing-api

# Start/stop/restart service
sudo systemctl start qr-processing-api
sudo systemctl stop qr-processing-api
sudo systemctl restart qr-processing-api

# View real-time logs
sudo journalctl -u qr-processing-api -f
```

### **After Code Changes**
```bash
# Restart service to apply changes
sudo systemctl restart qr-processing-api

# Verify service is running
curl -X GET "http://13.48.174.160:8002/health"
```

See `QR_SERVICE_MANAGEMENT.md` for complete service management documentation.

## 🛠️ Configuration

### Roboflow Workflow
The API uses a pre-configured Roboflow workflow named "qr-workflow" that:
- Detects QR code regions in images
- Returns cropped base64-encoded images of detected QR codes
- Provides confidence scores for each detection

### Image Processing Parameters
```python
# QR Decoding Methods (applied in sequence until successful)
METHODS = [
    "Direct grayscale",
    "Histogram equalization", 
    "Gaussian blur + contrast",
    "Morphological operations",
    "OTSU thresholding",
    "Adaptive thresholding + scaling"
]

# Temporary file settings
TEMP_FILE_SUFFIX = '.jpg'
CLEANUP_ON_COMPLETION = True
```

## 📊 Performance

### Typical Processing Times
- **Small folder (1-5 images)**: 10-30 seconds
- **Medium folder (10-20 images)**: 1-3 minutes
- **Large folder (50+ images)**: 5-15 minutes

### Success Rates
- **QR Detection**: ~90-95% (depends on image quality and QR code visibility)
- **QR Decoding**: ~60-80% (some detected QR codes may be too blurry/damaged to decode)
- **GPS Extraction**: ~95-99% (most drone images contain GPS data)

## 🐛 Troubleshooting

### Common Issues

#### 1. "libzbar0 not found" Error
```bash
sudo apt-get update
sudo apt-get install -y libzbar0
```

#### 2. S3 Access Denied
- Ensure AWS credentials are properly configured
- Check S3 bucket permissions
- Verify the S3 folder path exists

#### 3. Local Inference Server Not Available
- The API will automatically fallback to cloud API
- Check if inference server is running on port 9001
- No action needed - this is handled automatically

#### 4. High Memory Usage
- Large images are processed in sequence to manage memory
- Temporary files are cleaned up automatically
- Consider processing smaller batches for very large folders

### Debug Mode
Enable detailed logging by setting log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security

### AWS Credentials
- Uses IAM roles when available (recommended)
- Falls back to environment variables
- Never logs or exposes credentials

### API Security
- No authentication required (internal use)
- CORS enabled for cross-origin requests
- Input validation on S3 URLs

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8002/health
```

### Log Monitoring
The API provides detailed logging for:
- Image processing progress
- QR detection results  
- GPS extraction status
- Error conditions
- Performance metrics

### Example Log Output
```
INFO:__main__:Processing QR for folder: s3://bucket/folder/
INFO:__main__:Found 17 images in folder
INFO:__main__:Processing image: DJI_20250915113838_0007_V.jpeg
INFO:__main__:📍 GPS coordinates extracted: 24.949416, 55.513727
INFO:__main__:Using local inference server at localhost:9001
INFO:__main__:Found 5 QR code detections
INFO:__main__:✅ Decoded QR: https://qrto.org/9CmNYn...
```

## 🤝 Integration

### With Image Processing API
Both APIs can run simultaneously:
- **Image Processing API**: Port 8001
- **QR Processing API**: Port 8002

### Data Pipeline Integration
```python
# Example: Process images and QR codes in sequence
import requests

# 1. Process images for violations
image_response = requests.post("http://localhost:8001/process-folder", 
                              json={"s3_folder_url": folder_url})

# 2. Process same folder for QR codes
qr_response = requests.post("http://localhost:8002/process-qr",
                           json={"s3_folder_url": folder_url})

# 3. Combine results
combined_data = {
    "violations": image_response.json(),
    "qr_codes": qr_response.json()
}
```

## 📝 License

Internal use only. All rights reserved.

## 🆘 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review API logs for error details
3. Verify S3 access and Roboflow configuration
4. Test with a small folder first

---

**Last Updated**: September 16, 2025  
**API Version**: 1.0  
**Port**: 8002

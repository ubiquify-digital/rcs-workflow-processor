# Image Processing API - Complete Documentation

## Overview
A comprehensive FastAPI service for processing drone image folders with AI-powered object detection, GPS extraction, and advanced dumping violation analysis. Perfect for environmental monitoring, automated enforcement, and temporal change detection across multiple drone surveys.

## Features
- ✅ **S3 Integration**: Automatic folder discovery and processing
- ✅ **GPS Extraction**: Precise latitude/longitude/altitude from EXIF data
- ✅ **AI Object Detection**: Roboflow workflow integration with car/trash/license plate detection
- ✅ **Spatial Analysis**: GPS-based image comparison and clustering
- ✅ **Dumping Violation Detection**: Multi-folder analysis with license plate identification
- ✅ **Database Storage**: Supabase integration with optimized JSON
- ✅ **Real-time Processing**: Live status updates and progress tracking
- ✅ **Frontend Ready**: Rich APIs for web application integration

## Base URL
```
http://13.48.174.160:8001
```

## API Endpoints

### 🔍 **Discovery & Status**

#### `GET /s3-folders`
**Discover available S3 folders and their processing status**
- **Query Parameters:**
  - `bucket` (optional): S3 bucket name (default: "rcsstoragebucket")
  - `prefix` (optional): S3 prefix path (default: "djiimages/")
- **Response:**
```json
{
  "bucket": "rcsstoragebucket",
  "prefix": "djiimages/",
  "folders": [
    {
      "folder_name": "1",
      "s3_url": "s3://rcsstoragebucket/djiimages/1/",
      "is_processed": true,
      "processing_status": "completed",
      "task_id": "uuid",
      "total_images": 36,
      "processed_images": 36
    }
  ],
  "total_folders": 2,
  "processed_folders": 2
}
```

#### `GET /status/{task_id}`
**Get real-time processing status**
- **Response:**
```json
{
  "task_id": "uuid",
  "status": "processing",
  "progress": "Processing image 15/36",
  "processed_images": 15,
  "total_images": 36,
  "successful_images": 15,
  "failed_images": 0,
  "s3_output_folder_url": "s3://bucket/path/outputs/"
}
```

### ⚡ **Processing**

#### `POST /process-folder`
**Start processing an S3 folder**
- **Request Body:**
```json
{
  "s3_folder_url": "s3://rcsstoragebucket/djiimages/1/"
}
```
- **Response:**
```json
{
  "task_id": "uuid",
  "status": "queued",
  "message": "Image processing started"
}
```

### 📁 **Data Retrieval**

#### `GET /folders`
**List all processed folders**
- **Response:**
```json
{
  "folders": [
    {
      "id": "uuid",
      "task_id": "uuid", 
      "folder_name": "1",
      "s3_input_folder_url": "s3://bucket/path/",
      "s3_output_folder_url": "s3://bucket/path/outputs/",
      "status": "completed",
      "total_images": 36,
      "processed_images": 36,
      "successful_images": 36,
      "failed_images": 0,
      "created_at": "2025-09-08T14:41:31+00:00"
    }
  ]
}
```

#### `GET /folders/{task_id}`
**Get detailed folder information with all images**
- **Response:**
```json
{
  "folder_info": {
    "task_id": "uuid",
    "folder_name": "1",
    "status": "completed",
    "total_images": 36
  },
  "total_images": 36,
  "images": [
    {
      "id": "uuid",
      "filename": "DJI_20250903094325_0002_V.jpeg",
      "latitude": 24.949193,
      "longitude": 55.513019,
      "altitude": 117.08,
      "timestamp": "2025-09-03T09:43:25+00:00",
      "s3_input_url": "s3://rcsstoragebucket/djiimages/1/DJI_20250903094325_0002_V.jpeg",
      "s3_output_url": "s3://rcsstoragebucket/djiimages/1/outputs/DJI_20250903094325_0002_V.jpeg",
      "signed_input_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/DJI_20250903094325_0002_V.jpeg?AWSAccessKeyId=...&Signature=...&Expires=...",
      "signed_output_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/outputs/DJI_20250903094325_0002_V.jpeg?AWSAccessKeyId=...&Signature=...&Expires=...",
      "detections": [...] // AI detection results
    }
  ]
}
```

#### `GET /folders/{task_id}/summary`
**Get folder statistics and GPS bounds**
- **Response:**
```json
{
  "folder_info": {...},
  "statistics": {
    "total_images": 36,
    "images_with_gps": 36,
    "images_with_detections": 36,
    "gps_bounds": {
      "north": 24.949343,
      "south": 24.948564,
      "east": 55.514103,
      "west": 55.513019,
      "center": {
        "lat": 24.948933,
        "lon": 55.513736
      }
    },
    "time_range": {
      "earliest": "2025-09-03T09:43:25+00:00",
      "latest": "2025-09-03T09:45:46+00:00"
    }
  }
}
```

#### `GET /folders/{task_id}/images`
**Get all images for a folder**
- **Response:**
```json
{
  "images": [
    {
      "id": "uuid",
      "filename": "image.jpeg",
      "latitude": 24.949193,
      "longitude": 55.513019,
      "altitude": 117.08,
      "timestamp": "2025-09-03T09:43:25+00:00",
      "detections": [...],
      "s3_input_url": "s3://rcsstoragebucket/djiimages/1/image.jpeg",
      "s3_output_url": "s3://rcsstoragebucket/djiimages/1/outputs/image.jpeg",
      "signed_input_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=...",
      "signed_output_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/outputs/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=..."
    }
  ]
}
```

#### `GET /folders/{task_id}/images/paginated`
**Get paginated images with sorting**
- **Query Parameters:**
  - `page` (default: 1): Page number
  - `limit` (default: 20): Images per page
  - `sort_by` (default: "timestamp"): Sort field (timestamp, filename, latitude, longitude)
  - `order` (default: "asc"): Sort order (asc, desc)
- **Response:**
```json
{
  "images": [...],
  "pagination": {
    "current_page": 1,
    "total_pages": 5,
    "total_images": 100,
    "images_per_page": 20,
    "has_next": true,
    "has_previous": false
  },
  "sort": {
    "sort_by": "timestamp",
    "order": "asc"
  }
}
```

#### `GET /images/{image_id}`
**Get detailed information for a specific image**
- **Response:**
```json
{
  "image": {
    "id": "uuid",
    "filename": "image.jpeg",
    "latitude": 24.949193,
    "longitude": 55.513019,
    "altitude": 117.08,
    "timestamp": "2025-09-03T09:43:25+00:00",
    "detections": [...],
    "s3_input_url": "s3://rcsstoragebucket/djiimages/1/image.jpeg",
    "s3_output_url": "s3://rcsstoragebucket/djiimages/1/outputs/image.jpeg",
    "signed_input_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=...",
    "signed_output_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/outputs/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=..."
  }
}
```

### 🗺️ **Visualization Data**

#### `GET /folders/{task_id}/map-data`
**Get GPS coordinates for map visualization**
- **Response:**
```json
{
  "points": [
    {
      "id": "uuid",
      "filename": "image.jpeg",
      "lat": 24.949193,
      "lng": 55.513019,
      "timestamp": "2025-09-03T09:43:25+00:00",
      "s3_input_url": "s3://rcsstoragebucket/djiimages/1/image.jpeg",
      "s3_output_url": "s3://rcsstoragebucket/djiimages/1/outputs/image.jpeg",
      "signed_input_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=...",
      "signed_output_url": "https://rcsstoragebucket.s3.amazonaws.com/djiimages/1/outputs/image.jpeg?AWSAccessKeyId=...&Signature=...&Expires=..."
    }
  ],
  "bounds": {
    "north": 24.949343,
    "south": 24.948564,
    "east": 55.514103,
    "west": 55.513019,
    "center": {
      "lat": 24.948933,
      "lng": 55.513736
    }
  },
  "total_points": 36
}
```

#### `GET /folders/{task_id}/timeline`
**Get chronological timeline of images**
- **Response:**
```json
{
  "timeline": [
    {
      "id": "uuid",
      "filename": "image.jpeg",
      "timestamp": "2025-09-03T09:43:25+00:00",
      "has_gps": true,
      "gps": {
        "lat": 24.949193,
        "lng": 55.513019
      }
    }
  ],
  "time_info": {
    "start_time": "2025-09-03T09:43:25+00:00",
    "end_time": "2025-09-03T09:45:46+00:00"
  },
  "total_images": 36
}
```

### 🔍 **Analysis**

#### `GET /folders/{task_id}/detections-summary`
**Get detection analysis summary**
- **Response:**
```json
{
  "total_images": 36,
  "images_with_detections": 36,
  "total_detections": 36,
  "detection_types": {
    "car": 15,
    "person": 8,
    "building": 13
  },
  "sample_detections": [
    {
      "filename": "image.jpeg",
      "detection": {
        "class": "car",
        "confidence": 0.95,
        "bbox": [100, 100, 200, 200]
      }
    }
  ]
}
```

#### `POST /compare-folders`
**Compare images between folders by GPS location**
- **Request Body:**
```json
{
  "folder1_task_id": "uuid1",
  "folder2_task_id": "uuid2", 
  "distance_threshold_meters": 10.0
}
```
- **Response:**
```json
{
  "matches": [
    {
      "image1": {...}, // Full image data
      "image2": {...}, // Full image data
      "distance_meters": 2.05
    }
  ],
  "total_matches": 91,
  "folder1_images": 36,
  "folder2_images": 46,
  "distance_threshold_meters": 10.0
}
```

#### `POST /compare-folders-enriched`
**Enhanced comparison with detection change analysis**
Perfect for change detection scenarios like "car left trash", security monitoring, and environmental analysis.
- **Request Body:**
```json
{
  "folder1_task_id": "uuid1",
  "folder2_task_id": "uuid2", 
  "distance_threshold_meters": 5.0
}
```
- **Response:**
```json
{
  "matches": [
    {
      "image1": {...}, // Full image data with signed URLs
      "image2": {...}, // Full image data with signed URLs
      "distance_meters": 0.03,
      "time_difference_minutes": 8.9,
      "detection_changes": [
        {
          "object_type": "car",
          "change_type": "disappeared",
          "before_count": 1,
          "after_count": 0
        },
        {
          "object_type": "trash",
          "change_type": "appeared", 
          "before_count": 0,
          "after_count": 1
        }
      ],
      "change_summary": "1 car(s) disappeared; 1 trash(s) appeared"
    }
  ],
  "total_matches": 34,
  "folder1_images": 36,
  "folder2_images": 46,
  "distance_threshold_meters": 5.0,
  "change_statistics": {
    "objects_appeared": 4,
    "objects_disappeared": 26,
    "count_changes": 0,
    "no_changes": 73
  }
}
```

#### `POST /analyze-multifolder-dumping`
**🚗🗑️ Advanced Multi-Folder Dumping Violation Analysis**
Analyze multiple drone runs across time to detect car dumping violations with spatial clustering, chronological sorting, and license plate identification.

Perfect for:
- **Environmental monitoring** across multiple drone surveys
- **Automated violation detection** with license plate identification
- **Evidence collection** with before/after frame pairs
- **Temporal analysis** of dumping patterns

- **Request Body:**
```json
{
  "task_ids": ["uuid1", "uuid2", "uuid3", "..."],
  "distance_threshold_meters": 2.0,
  "min_time_gap_minutes": 5.0
}
```

- **Response:**
```json
{
  "violations": [
    {
      "cluster_id": 8,
      "violation_type": "car_left_trash",
      "location": {
        "lat": 24.948959235,
        "lon": 55.51321097
      },
      "before_frame": {
        "id": "uuid",
        "filename": "DJI_20250903094346_0009_V.jpeg",
        "folder_name": "1",
        "timestamp": "2025-09-03T09:43:46+00:00",
        "latitude": 24.94895819,
        "longitude": 55.51321183,
        "signed_input_url": "https://s3.amazonaws.com/...",
        "signed_output_url": "https://s3.amazonaws.com/...",
        "detections": [...]
      },
      "after_frame": {
        "id": "uuid",
        "filename": "DJI_20250903095238_0009_V.jpeg", 
        "folder_name": "2",
        "timestamp": "2025-09-03T09:52:38+00:00",
        "latitude": 24.94896144,
        "longitude": 55.51321456,
        "signed_input_url": "https://s3.amazonaws.com/...",
        "signed_output_url": "https://s3.amazonaws.com/...",
        "detections": [...]
      },
      "before_folder": "1",
      "after_folder": "2", 
      "vehicle_plate": "H 68773",
      "time_difference_minutes": 8.9,
      "distance_meters": 0.29,
      "trash_count": 3,
      "description": "🚗🗑️ DUMPING DETECTED: Car [H 68773] left 3 trash item(s) behind"
    }
  ],
  "total_violations": 12,
  "clusters_analyzed": 142,
  "total_frames": 245,
  "folders_analyzed": ["1", "2", "8SEP Run 1", "8SEP Run 2"],
  "analysis_summary": {
    "car_arrived_dumped": 8,
    "car_left_trash": 4
  }
}
```

**Key Features:**
- **🎯 Spatial Clustering**: Groups images by GPS location within configurable radius
- **⏰ Chronological Analysis**: Sorts frames by timestamp within each cluster
- **🚗 License Plate Detection**: Automatically extracts vehicle identification
- **📏 Precise Measurements**: GPS distance between before/after frames
- **🔗 Evidence Links**: Signed URLs for both original and processed images
- **📊 Pattern Recognition**: Detects 4 types of dumping violations:
  - `car_left_trash`: Car disappeared, trash remained
  - `car_dumped_trash`: Car present, additional trash appeared
  - `car_arrived_dumped`: Car arrived with trash at clean location
  - `additional_dumping`: More trash added to existing pile

**Violation Types Explained:**
- **car_left_trash**: Vehicle was present initially, then left, leaving trash behind
- **car_dumped_trash**: Vehicle was present in both frames, but dumped additional trash
- **car_arrived_dumped**: Clean area initially, then vehicle arrived and dumped trash
- **additional_dumping**: Existing trash pile, vehicle added more items

### 📥 **Export**

#### `POST /folders/{task_id}/generate-srt`
**Generate SRT subtitle file with GPS coordinates for video overlay**
Perfect for adding GPS data overlays to drone videos at any frame rate.
- **Request Body:**
```json
{
  "task_id": "uuid",
  "video_duration_seconds": 196,
  "fps": 1,
  "font_size": 28,
  "coordinate_precision": 6
}
```
- **Response:**
```json
{
  "srt_content": "1\n00:00:00,000 --> 00:00:01,000\n<font size=\"28\">GPS: Lat 24.949193, Lon 55.513019</font>\n\n2\n00:00:01,000 --> 00:00:02,000\n<font size=\"28\">GPS: Lat 24.949184, Lon 55.513026</font>\n\n...",
  "total_entries": 196,
  "duration_seconds": 196,
  "gps_range": {
    "latitude": {
      "min": 24.948564,
      "max": 24.949343
    },
    "longitude": {
      "min": 55.513019,
      "max": 55.514103
    }
  },
  "images_used": 36,
  "interpolated_points": 160
}
```

#### `GET /folders/{task_id}/export`
**Export complete folder data**
- **Query Parameters:**
  - `format` (default: "json"): Export format
- **Response:**
```json
{
  "folder_info": {...},
  "images": [...],
  "export_metadata": {
    "exported_at": "2025-09-08T15:30:00+00:00",
    "total_images": 36,
    "format": "json"
  }
}
```

### 🛠️ **Utility**

#### `GET /health`
**API health check**
- **Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-08T15:30:00+00:00"
}
```

#### `GET /`
**API root endpoint**
- **Response:**
```json
{
  "message": "Image Processing API",
  "version": "1.0.0"
}
```

## Frontend Integration Examples

### Dashboard Overview
```javascript
// Get all S3 folders with processing status
const folders = await fetch('http://13.48.174.160:8001/s3-folders');

// Start processing unprocessed folder
await fetch('http://13.48.174.160:8001/process-folder', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    s3_folder_url: 's3://rcsstoragebucket/djiimages/3/'
  })
});
```

### Map Visualization
```javascript
// Get GPS data for map
const mapData = await fetch('http://13.48.174.160:8001/folders/{task_id}/map-data');
const {points, bounds} = await mapData.json();

// Initialize map with bounds
const map = new google.maps.Map(element, {
  center: bounds.center,
  zoom: calculateZoom(bounds)
});

// Add markers for each image
points.forEach(point => {
  new google.maps.Marker({
    position: {lat: point.lat, lng: point.lng},
    map: map,
    title: point.filename
  });
});
```

### Paginated Gallery
```javascript
// Load paginated images
const images = await fetch(
  'http://13.48.174.160:8001/folders/{task_id}/images/paginated?page=1&limit=20&sort_by=timestamp'
);
const {images: imageList, pagination} = await images.json();

// Render pagination controls
renderPagination(pagination);
```

### Enhanced Change Detection
```javascript
// Compare two drone flights with change analysis
const changeAnalysis = await fetch('http://13.48.174.160:8001/compare-folders-enriched', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    folder1_task_id: 'morning-flight-uuid',
    folder2_task_id: 'evening-flight-uuid',
    distance_threshold_meters: 5.0
  })
});

const {matches, change_statistics} = await changeAnalysis.json();

// Display change statistics
console.log(`Objects appeared: ${change_statistics.objects_appeared}`);
console.log(`Objects disappeared: ${change_statistics.objects_disappeared}`);

// Process matches with changes
const significantChanges = matches.filter(match => match.detection_changes.length > 0);
significantChanges.forEach(match => {
  console.log(`Location: ${match.distance_meters}m apart`);
  console.log(`Time difference: ${match.time_difference_minutes} minutes`);
  console.log(`Changes: ${match.change_summary}`);
  
  // Display before/after images
  showImageComparison(match.image1.signed_input_url, match.image2.signed_input_url);
});
```

### Multi-Folder Dumping Violation Analysis
```javascript
// Analyze multiple drone runs for dumping violations
const dumpingAnalysis = await fetch('http://13.48.174.160:8001/analyze-multifolder-dumping', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    task_ids: [
      'morning-patrol-uuid',
      'afternoon-patrol-uuid', 
      'evening-patrol-uuid'
    ],
    distance_threshold_meters: 2.0,  // 2m clustering radius
    min_time_gap_minutes: 10.0       // Minimum 10 minutes between violations
  })
});

const {violations, total_violations, analysis_summary, folders_analyzed} = await dumpingAnalysis.json();

// Display violation summary
console.log(`🚨 Found ${total_violations} dumping violations across ${folders_analyzed.length} drone runs`);
console.log('Violation breakdown:', analysis_summary);

// Process each violation with evidence
violations.forEach(violation => {
  console.log(`\n🚗🗑️ ${violation.description}`);
  console.log(`📍 Location: ${violation.location.lat}, ${violation.location.lon}`);
  console.log(`⏰ Time gap: ${violation.time_difference_minutes} minutes`);
  console.log(`📏 GPS precision: ${violation.distance_meters}m between frames`);
  console.log(`📁 Evidence: ${violation.before_folder} → ${violation.after_folder}`);
  
  if (violation.vehicle_plate) {
    console.log(`🚗 Vehicle: ${violation.vehicle_plate}`);
  }
  
  // Display evidence images
  showViolationEvidence({
    beforeImage: violation.before_frame.signed_input_url,
    afterImage: violation.after_frame.signed_input_url,
    beforeProcessed: violation.before_frame.signed_output_url,
    afterProcessed: violation.after_frame.signed_output_url,
    description: violation.description
  });
  
  // Generate violation report
  generateViolationReport({
    violationType: violation.violation_type,
    vehiclePlate: violation.vehicle_plate,
    location: violation.location,
    timestamp: violation.after_frame.timestamp,
    evidenceUrls: [
      violation.before_frame.signed_input_url,
      violation.after_frame.signed_input_url
    ]
  });
});
```

### SRT Generation for Video Overlays
```javascript
// Generate GPS subtitle overlay for drone video
const srtResponse = await fetch('http://13.48.174.160:8001/folders/{task_id}/generate-srt', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    task_id: 'drone-flight-uuid',
    video_duration_seconds: 300,  // 5 minute video
    fps: 2,                       // 2 subtitles per second for smooth overlay
    font_size: 32,
    coordinate_precision: 6
  })
});

const {srt_content, total_entries, gps_range, images_used} = await srtResponse.json();

// Save SRT file for video editor
const blob = new Blob([srt_content], { type: 'text/plain' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'drone_gps_overlay.srt';
a.click();

console.log(`Generated ${total_entries} GPS subtitles from ${images_used} images`);
console.log(`GPS Range: ${gps_range.latitude.min} to ${gps_range.latitude.max} lat`);
```

## Error Responses
All endpoints return consistent error responses:
```json
{
  "detail": "Error description"
}
```

Common HTTP status codes:
- `200`: Success
- `404`: Resource not found
- `422`: Validation error
- `500`: Internal server error

## Rate Limiting
Currently no rate limiting implemented. Consider implementing for production use.

## Signed URLs
All image responses now include signed URLs for secure access to S3 objects:
- `signed_input_url`: Pre-signed URL for the original image (expires in 1 hour)
- `signed_output_url`: Pre-signed URL for the processed image (expires in 1 hour)

These URLs can be used directly in web applications without requiring AWS credentials.

## Enhanced Change Detection
The enriched compare endpoint (`/compare-folders-enriched`) provides intelligent analysis of detection changes between matched images:

### **Detection Change Types:**
- **`appeared`**: New objects detected in the second image
- **`disappeared`**: Objects present in first image but missing in second
- **`count_changed`**: Same object type but different quantities

### **Supported Object Types:**
- **Cars & Vehicles**: From specialized car detection models
- **License Plates**: From license plate recognition models  
- **Generic Objects**: Any object detected by OpenAI models (people, trash, packages, equipment, animals, etc.)

### **Use Cases:**
- **🚗➡️🗑️ Environmental Monitoring**: "Car left trash" scenarios
- **📦 Delivery Tracking**: Package appearance/disappearance
- **🚧 Construction Progress**: Equipment and material changes
- **🔒 Security Analysis**: Unauthorized object placement
- **📊 Traffic Studies**: Vehicle pattern changes
- **🏗️ Site Monitoring**: Infrastructure development tracking

### **Distance Precision:**
- **1m**: Ultra-precise matching (34 matches) - identical viewpoints
- **5m**: High-precision analysis (91 matches) - close proximity  
- **10m**: Standard comparison (170 matches) - general area
- **50m**: Broad coverage (745 matches) - wide area analysis

## SRT Subtitle Generation
Generate SRT subtitle files with GPS coordinates for video overlays, perfect for drone footage:

### **Features:**
- **Configurable Frame Rate**: 1fps (standard) to 60fps for smooth overlays
- **Smart Interpolation**: Linear interpolation between GPS points for missing timestamps
- **Customizable Formatting**: Font size, coordinate precision, and styling
- **Time-based Alignment**: Uses image timestamps for accurate video synchronization

### **Parameters:**
- **`video_duration_seconds`**: Total video length (default: 196s)
- **`fps`**: Subtitles per second (default: 1fps)
- **`font_size`**: HTML font size for subtitles (default: 28)
- **`coordinate_precision`**: Decimal places for GPS coordinates (default: 6)

### **Use Cases:**
- **🎥 Drone Video Overlays**: Add GPS coordinates to flight footage
- **📍 Location Documentation**: Track exact positions during video recording
- **🗺️ Mapping Projects**: Correlate video content with precise locations
- **📊 Analysis Tools**: Provide spatial context for video analysis
- **🚁 Flight Path Visualization**: Show GPS trajectory during drone flights

## Authentication
Currently no authentication required. Consider implementing API keys for production use.

## Data Models

### Image Record
```json
{
  "id": "uuid",
  "task_id": "uuid", 
  "filename": "string",
  "s3_input_url": "string",
  "s3_output_url": "string",
  "signed_input_url": "string",
  "signed_output_url": "string",
  "timestamp": "ISO datetime",
  "latitude": "decimal",
  "longitude": "decimal", 
  "altitude": "decimal",
  "detections": "JSONB",
  "processing_status": "string",
  "processed_at": "ISO datetime"
}
```

### Folder Record
```json
{
  "id": "uuid",
  "task_id": "uuid",
  "folder_name": "string",
  "s3_input_folder_url": "string", 
  "s3_output_folder_url": "string",
  "status": "string",
  "total_images": "integer",
  "processed_images": "integer",
  "successful_images": "integer",
  "failed_images": "integer",
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime"
}
```

## Performance Notes
- **Pagination**: Use paginated endpoints for large datasets
- **Map Data**: Lightweight endpoint optimized for mapping
- **Caching**: Consider implementing Redis caching for frequently accessed data
- **Database**: Supabase handles connection pooling and optimization

## Deployment
Service runs as systemd service on Ubuntu with auto-restart capabilities.

## Support
For technical support or feature requests, contact the development team.

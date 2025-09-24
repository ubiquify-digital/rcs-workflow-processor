# VIN Tracking API Documentation

## Overview

The VIN Tracking API provides comprehensive tracking and management of Vehicle Identification Numbers (VINs) detected from drone imagery. It offers a complete dashboard view with VIN history, location tracking, and map visualization.

## Base URL

```
https://inference.rcs.exponent-ts.com/qr-api
```

## Endpoints

### 1. List All VINs

**GET** `/vins`

Returns all unique VINs with summary information including spottings, folders, and AI descriptions.

#### Response

```json
{
  "vins": [
    {
      "vin": "1HGBH41JXMN109188",
      "total_spottings": 12,
      "first_spotted": "2025-09-23T13:02:25.877053Z",
      "last_spotted": "2025-09-23T13:06:03.777744Z",
      "folders": ["Run September 22 5:55 PM", "Run September 22 5:40 PM", "Run September 22 6:07 PM"],
      "latest_location": {
        "latitude": 24.94949742,
        "longitude": 55.51372136
      },
      "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 1991\n- Body: Sedan\n- Engine: 2.2L 4-cylinder\n- Assembly Plant: Marysville, Ohio"
    }
  ],
  "total_vins": 5
}
```

### 2. Get VIN History

**GET** `/vins/{vin}/history`

Returns detailed history for a specific VIN showing all spottings with timestamps, locations, and images.

#### Response

```json
{
  "vin": "1HGBH41JXMN109188",
  "history": [
    {
      "vin": "1HGBH41JXMN109188",
      "folder_name": "Run September 22 6:07 PM",
      "folder_s3_url": "s3://rcsstoragebucket/qr_sync/...",
      "image_filename": "DJI_20250922180737_0007_V.jpeg",
      "image_url": "https://signed-url-for-image",
      "spotted_at": "2025-09-23T13:02:25.877053Z",
      "latitude": 24.94947483,
      "longitude": 55.5137405,
      "confidence": 0.878923237323761
    }
  ],
  "total_spottings": 12
}
```

### 3. Get VIN Summary

**GET** `/vins/{vin}`

Returns summary information for a specific VIN.

#### Response

```json
{
  "vin": "1HGBH41JXMN109188",
  "total_spottings": 12,
  "first_spotted": "2025-09-23T13:02:25.877053Z",
  "last_spotted": "2025-09-23T13:06:03.777744Z",
  "folders": ["Run September 22 5:55 PM", "Run September 22 5:40 PM", "Run September 22 6:07 PM"],
  "latest_location": {
    "latitude": 24.94949742,
    "longitude": 55.51372136
  },
  "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 1991\n- Body: Sedan\n- Engine: 2.2L 4-cylinder\n- Assembly Plant: Marysville, Ohio"
}
```

### 4. Get Latest Folder Map

**GET** `/map/latest`

Returns map data for the latest processed folder with VIN locations and coordinates.

#### Response

```json
{
  "folder_name": "Run September 22 5:40 PM",
  "folder_s3_url": "s3://rcsstoragebucket/qr_sync/...",
  "total_images": 42,
  "total_vins": 5,
  "points": [
    {
      "vin": "1HGBH41JXMN109188",
      "latitude": 24.94949233,
      "longitude": 55.51372772,
      "image_url": "https://signed-url-for-image",
      "spotted_at": "2025-09-23T13:05:42.666837Z",
      "confidence": 0.9006134271621704
    }
  ],
  "bounds": {
    "north": 24.94949742,
    "south": 24.94937058,
    "east": 55.51381592,
    "west": 55.51372136
  }
}
```

### 5. VIN Dashboard

**GET** `/dashboard`

Returns comprehensive dashboard data combining all VINs and latest map information.

#### Response

```json
{
  "vins": [
    {
      "vin": "1HGBH41JXMN109188",
      "total_spottings": 12,
      "first_spotted": "2025-09-23T13:02:25.877053Z",
      "last_spotted": "2025-09-23T13:06:03.777744Z",
      "folders": ["Run September 22 5:55 PM", "Run September 22 5:40 PM", "Run September 22 6:07 PM"],
      "latest_location": {
        "latitude": 24.94949742,
        "longitude": 55.51372136
      },
      "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 1991\n- Body: Sedan\n- Engine: 2.2L 4-cylinder\n- Assembly Plant: Marysville, Ohio"
    }
  ],
  "latest_map": {
    "folder_name": "Run September 22 5:40 PM",
    "folder_s3_url": "s3://rcsstoragebucket/qr_sync/...",
    "total_images": 42,
    "total_vins": 5,
    "points": [...],
    "bounds": {...}
  },
  "total_vins": 5,
  "total_folders": 7
}
```

## Data Models

### VINSummary

```typescript
interface VINSummary {
  vin: string;
  total_spottings: number;
  first_spotted: string; // ISO datetime
  last_spotted: string; // ISO datetime
  folders: string[];
  latest_location?: {
    latitude: number;
    longitude: number;
  };
  ai_description?: string;
}
```

### VINHistory

```typescript
interface VINHistory {
  vin: string;
  folder_name: string;
  folder_s3_url: string;
  image_filename: string;
  image_url: string;
  spotted_at: string; // ISO datetime
  latitude?: number;
  longitude?: number;
  confidence: number;
}
```

### MapPoint

```typescript
interface MapPoint {
  vin: string;
  latitude: number;
  longitude: number;
  image_url: string;
  spotted_at: string; // ISO datetime
  confidence: number;
}
```

### MapData

```typescript
interface MapData {
  folder_name: string;
  folder_s3_url: string;
  total_images: number;
  total_vins: number;
  points: MapPoint[];
  bounds?: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
}
```

## Features

### VIN Tracking
- **Complete History**: Track when and where each VIN was spotted
- **Folder Association**: Know which drone runs contained each VIN
- **Location Tracking**: GPS coordinates for each VIN spotting
- **Confidence Scores**: QR code detection confidence levels

### AI Integration
- **Vehicle Descriptions**: AI-generated vehicle information from VINs
- **Make/Model/Year**: Detailed vehicle specifications
- **Assembly Plant**: Manufacturing location information

### Map Visualization
- **Latest Folder Map**: Show VIN locations on latest processed folder
- **GPS Coordinates**: Precise location data for mapping
- **Bounds Calculation**: Automatic map bounds for optimal viewing
- **Image Links**: Direct links to drone images showing VINs

### Dashboard Features
- **VIN Sidebar**: Complete list of all detected VINs
- **History Timeline**: Chronological VIN spotting history
- **Map Integration**: Visual representation of VIN locations
- **Statistics**: Total VINs, folders, and spottings

## Usage Examples

### Get All VINs for Sidebar

```bash
curl "https://inference.rcs.exponent-ts.com/qr-api/vins"
```

### Get VIN History for Timeline

```bash
curl "https://inference.rcs.exponent-ts.com/qr-api/vins/1HGBH41JXMN109188/history"
```

### Get Latest Map Data

```bash
curl "https://inference.rcs.exponent-ts.com/qr-api/map/latest"
```

### Get Complete Dashboard

```bash
curl "https://inference.rcs.exponent-ts.com/qr-api/dashboard"
```

## Frontend Integration

### React Component Example

```typescript
interface VINDashboardProps {
  vins: VINSummary[];
  latestMap: MapData;
  totalVins: number;
  totalFolders: number;
}

const VINDashboard: React.FC<VINDashboardProps> = ({ vins, latestMap, totalVins, totalFolders }) => {
  return (
    <div className="vin-dashboard">
      <div className="sidebar">
        <h3>VINs ({totalVins})</h3>
        {vins.map(vin => (
          <div key={vin.vin} className="vin-item">
            <div className="vin-number">{vin.vin}</div>
            <div className="vin-info">
              <div>Spottings: {vin.total_spottings}</div>
              <div>Last seen: {new Date(vin.last_spotted).toLocaleString()}</div>
              <div>Folders: {vin.folders.length}</div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="main-content">
        <div className="map-container">
          <h3>Latest Folder: {latestMap.folder_name}</h3>
          <div className="map-stats">
            <div>Images: {latestMap.total_images}</div>
            <div>VINs: {latestMap.total_vins}</div>
          </div>
          {/* Map component with latestMap.points */}
        </div>
      </div>
    </div>
  );
};
```

### Map Integration

```typescript
// Using Leaflet or similar mapping library
const MapComponent: React.FC<{ points: MapPoint[] }> = ({ points }) => {
  return (
    <MapContainer bounds={calculateBounds(points)}>
      {points.map((point, index) => (
        <Marker
          key={index}
          position={[point.latitude, point.longitude]}
          popup={
            <div>
              <div>VIN: {point.vin}</div>
              <div>Confidence: {(point.confidence * 100).toFixed(1)}%</div>
              <div>Time: {new Date(point.spotted_at).toLocaleString()}</div>
              <img src={point.image_url} alt="VIN Image" width="200" />
            </div>
          }
        />
      ))}
    </MapContainer>
  );
};
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **404**: VIN not found or no completed folders
- **500**: Server error

Error responses include detailed error messages:

```json
{
  "detail": "VIN 1HGBH41JXMN109188 not found"
}
```

## Performance Notes

- VIN data is cached in the database for fast retrieval
- Signed URLs are generated fresh for each request
- Large datasets are paginated for optimal performance
- Map bounds are calculated automatically for efficient rendering

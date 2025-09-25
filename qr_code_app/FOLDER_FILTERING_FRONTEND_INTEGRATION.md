# Folder Filtering Frontend Integration Guide

## Overview
The VIN tracking API now supports optional folder filtering for all endpoints. This allows the frontend to filter VINs and maps by specific drone runs/folders.

## New API Endpoints

### 1. List Available Folders
**GET** `/folders`

Returns all available completed folders with metadata.

**Response:**
```json
{
  "folders": [
    {
      "folder_name": "Run September 22 6:07 PM",
      "s3_url": "https://bucket.s3.amazonaws.com/folder/",
      "total_images": 150,
      "created_at": "2025-09-22T18:07:00Z",
      "status": "completed"
    }
  ],
  "total": 1
}
```

### 2. Updated Dashboard Endpoint
**GET** `/dashboard?folders=folder1,folder2`

Optional `folders` parameter (comma-separated list of folder names).

**Examples:**
- `/dashboard` - All folders (default behavior)
- `/dashboard?folders=Run September 22 6:07 PM` - Single folder
- `/dashboard?folders=Run September 22 6:07 PM,Run September 22 5:30 PM` - Multiple folders

### 3. Updated VINs Endpoint
**GET** `/vins?folders=folder1,folder2`

Optional `folders` parameter (comma-separated list of folder names).

**Examples:**
- `/vins` - All VINs from all folders
- `/vins?folders=Run September 22 6:07 PM` - VINs from specific folder only

## Frontend Integration

### TypeScript Interfaces

```typescript
interface Folder {
  folder_name: string;
  s3_url: string;
  total_images: number;
  created_at: string;
  status: string;
}

interface FoldersResponse {
  folders: Folder[];
  total: number;
}

interface VINSummary {
  vin: string;
  total_spottings: number;
  first_spotted: string;
  last_spotted: string;
  folders: string[];
  latest_location?: {
    latitude: number;
    longitude: number;
  };
  ai_description?: string;
}

interface VINListResponse {
  vins: VINSummary[];
  total_vins: number;
}

interface MapPoint {
  vin: string;
  latitude: number;
  longitude: number;
  image_url: string;
  spotted_at: string;
  confidence: number;
}

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

interface VINDashboardResponse {
  vins: VINSummary[];
  latest_map?: MapData;
  total_vins: number;
  total_folders: number;
}
```

### API Service Functions

```typescript
// API base URL
const API_BASE = 'https://inference.rcs.exponent-ts.com/qr-api';

// Get all available folders
export const getFolders = async (): Promise<FoldersResponse> => {
  const response = await fetch(`${API_BASE}/folders`);
  if (!response.ok) throw new Error('Failed to fetch folders');
  return response.json();
};

// Get VINs with optional folder filtering
export const getVINs = async (folders?: string[]): Promise<VINListResponse> => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folders', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/vins?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch VINs');
  return response.json();
};

// Get dashboard with optional folder filtering
export const getDashboard = async (folders?: string[]): Promise<VINDashboardResponse> => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folders', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/dashboard?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch dashboard');
  return response.json();
};

// Get VIN history (existing endpoint, no changes)
export const getVINHistory = async (vin: string) => {
  const response = await fetch(`${API_BASE}/vins/${vin}/history`);
  if (!response.ok) throw new Error('Failed to fetch VIN history');
  return response.json();
};

// Get VIN movement path (existing endpoint, no changes)
export const getVINMovement = async (vin: string) => {
  const response = await fetch(`${API_BASE}/vins/${vin}/movement`);
  if (!response.ok) throw new Error('Failed to fetch VIN movement');
  return response.json();
};
```

### React Component Example

```typescript
import React, { useState, useEffect } from 'react';

interface FolderSelectorProps {
  selectedFolders: string[];
  onFoldersChange: (folders: string[]) => void;
}

const FolderSelector: React.FC<FolderSelectorProps> = ({ 
  selectedFolders, 
  onFoldersChange 
}) => {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFolders = async () => {
      try {
        const response = await getFolders();
        setFolders(response.folders);
      } catch (error) {
        console.error('Error fetching folders:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchFolders();
  }, []);

  const handleFolderToggle = (folderName: string) => {
    if (selectedFolders.includes(folderName)) {
      onFoldersChange(selectedFolders.filter(f => f !== folderName));
    } else {
      onFoldersChange([...selectedFolders, folderName]);
    }
  };

  const handleSelectAll = () => {
    onFoldersChange(folders.map(f => f.folder_name));
  };

  const handleSelectNone = () => {
    onFoldersChange([]);
  };

  if (loading) return <div>Loading folders...</div>;

  return (
    <div className="folder-selector">
      <div className="folder-controls">
        <button onClick={handleSelectAll}>Select All</button>
        <button onClick={handleSelectNone}>Select None</button>
        <span>{selectedFolders.length} of {folders.length} folders selected</span>
      </div>
      
      <div className="folder-list">
        {folders.map(folder => (
          <label key={folder.folder_name} className="folder-item">
            <input
              type="checkbox"
              checked={selectedFolders.includes(folder.folder_name)}
              onChange={() => handleFolderToggle(folder.folder_name)}
            />
            <div className="folder-info">
              <div className="folder-name">{folder.folder_name}</div>
              <div className="folder-meta">
                {folder.total_images} images • {new Date(folder.created_at).toLocaleDateString()}
              </div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
};

// Main Dashboard Component
const VINDashboard: React.FC = () => {
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [dashboardData, setDashboardData] = useState<VINDashboardResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchDashboard = async () => {
      setLoading(true);
      try {
        const data = await getDashboard(selectedFolders.length > 0 ? selectedFolders : undefined);
        setDashboardData(data);
      } catch (error) {
        console.error('Error fetching dashboard:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboard();
  }, [selectedFolders]);

  return (
    <div className="vin-dashboard">
      <div className="dashboard-header">
        <h1>VIN Tracking Dashboard</h1>
        <FolderSelector 
          selectedFolders={selectedFolders}
          onFoldersChange={setSelectedFolders}
        />
      </div>
      
      {loading ? (
        <div>Loading dashboard...</div>
      ) : dashboardData ? (
        <div className="dashboard-content">
          <div className="stats">
            <div className="stat">
              <span className="stat-label">Total VINs:</span>
              <span className="stat-value">{dashboardData.total_vins}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Total Folders:</span>
              <span className="stat-value">{dashboardData.total_folders}</span>
            </div>
          </div>
          
          <div className="dashboard-grid">
            <div className="vins-panel">
              <h2>VINs ({dashboardData.vins.length})</h2>
              <div className="vin-list">
                {dashboardData.vins.map(vin => (
                  <div key={vin.vin} className="vin-item">
                    <div className="vin-number">{vin.vin}</div>
                    <div className="vin-info">
                      <div>{vin.total_spottings} spottings</div>
                      <div>Last seen: {new Date(vin.last_spotted).toLocaleString()}</div>
                      {vin.ai_description && (
                        <div className="ai-description">{vin.ai_description}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {dashboardData.latest_map && (
              <div className="map-panel">
                <h2>Latest Map: {dashboardData.latest_map.folder_name}</h2>
                <div className="map-info">
                  <div>{dashboardData.latest_map.total_images} images</div>
                  <div>{dashboardData.latest_map.total_vins} VINs detected</div>
                </div>
                {/* Render your map component here */}
                <div className="map-container">
                  {/* Map implementation */}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div>No data available</div>
      )}
    </div>
  );
};

export default VINDashboard;
```

### CSS Styling

```css
.folder-selector {
  background: #f5f5f5;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.folder-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
}

.folder-controls button {
  padding: 0.5rem 1rem;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
}

.folder-controls button:hover {
  background: #f0f0f0;
}

.folder-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.folder-item {
  display: flex;
  align-items: center;
  padding: 0.5rem;
  border-bottom: 1px solid #eee;
  cursor: pointer;
}

.folder-item:hover {
  background: #f9f9f9;
}

.folder-item:last-child {
  border-bottom: none;
}

.folder-item input[type="checkbox"] {
  margin-right: 0.5rem;
}

.folder-info {
  flex: 1;
}

.folder-name {
  font-weight: 500;
  color: #333;
}

.folder-meta {
  font-size: 0.875rem;
  color: #666;
  margin-top: 0.25rem;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

.vins-panel, .map-panel {
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 1rem;
}

.vin-item {
  padding: 0.75rem;
  border: 1px solid #eee;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.vin-number {
  font-family: monospace;
  font-weight: bold;
  color: #2563eb;
}

.ai-description {
  font-size: 0.875rem;
  color: #666;
  margin-top: 0.25rem;
  font-style: italic;
}

.stats {
  display: flex;
  gap: 2rem;
  margin-bottom: 1rem;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-label {
  font-size: 0.875rem;
  color: #666;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #333;
}
```

## Usage Examples

### 1. Filter by Single Folder
```typescript
// Get VINs from only the latest folder
const latestFolder = "Run September 22 6:07 PM";
const vins = await getVINs([latestFolder]);
```

### 2. Filter by Multiple Folders
```typescript
// Get VINs from specific drone runs
const selectedFolders = [
  "Run September 22 6:07 PM",
  "Run September 22 5:30 PM"
];
const vins = await getVINs(selectedFolders);
```

### 3. Get All VINs (No Filtering)
```typescript
// Get all VINs from all folders
const vins = await getVINs();
```

### 4. Dashboard with Folder Selection
```typescript
// Dashboard with folder filtering
const dashboard = await getDashboard(["Run September 22 6:07 PM"]);
```

## Benefits

1. **Performance**: Filter by specific folders to reduce data load
2. **User Experience**: Allow users to focus on specific drone runs
3. **Analysis**: Compare VINs across different time periods
4. **Flexibility**: Easy to implement "Select All" or "Select None" functionality

## Backward Compatibility

- All existing endpoints work without changes when no `folders` parameter is provided
- Default behavior remains the same (all folders)
- No breaking changes to existing frontend implementations

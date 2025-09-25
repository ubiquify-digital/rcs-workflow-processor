# Folder Filtering - Minimal Frontend Guide

## Overview
All VIN tracking endpoints now support optional folder filtering via the `folder_filter` query parameter.

## Updated Endpoints

### 1. VIN History with Filtering
**GET** `/vins/{vin}/history?folder_filter=folder1,folder2`

**Example:**
```typescript
// Get history for specific folders only
const history = await fetch(
  `${API_BASE}/vins/1GNSK8KD3NR150647/history?folder_filter=Run%20September%2022%206:07%20PM`
);
```

### 2. VIN Movement with Filtering
**GET** `/vins/{vin}/movement?folder_filter=folder1,folder2`

**Example:**
```typescript
// Get movement path for specific folders only
const movement = await fetch(
  `${API_BASE}/vins/1GNSK8KD3NR150647/movement?folder_filter=Run%20September%2022%206:07%20PM`
);
```

### 3. Dashboard with Filtering (Already Implemented)
**GET** `/dashboard?folders=folder1,folder2`

### 4. VINs List with Filtering (Already Implemented)
**GET** `/vins?folders=folder1,folder2`

## Frontend Implementation

### API Service Functions

```typescript
const API_BASE = 'https://inference.rcs.exponent-ts.com/qr-api';

// Get VIN history with optional folder filtering
export const getVINHistory = async (vin: string, folders?: string[]) => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folder_filter', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/vins/${vin}/history?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch VIN history');
  return response.json();
};

// Get VIN movement with optional folder filtering
export const getVINMovement = async (vin: string, folders?: string[]) => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folder_filter', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/vins/${vin}/movement?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch VIN movement');
  return response.json();
};

// Get dashboard with optional folder filtering (existing)
export const getDashboard = async (folders?: string[]) => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folders', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/dashboard?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch dashboard');
  return response.json();
};

// Get VINs with optional folder filtering (existing)
export const getVINs = async (folders?: string[]) => {
  const params = new URLSearchParams();
  if (folders && folders.length > 0) {
    params.append('folders', folders.join(','));
  }
  
  const response = await fetch(`${API_BASE}/vins?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch VINs');
  return response.json();
};
```

### React Component Example

```typescript
import React, { useState, useEffect } from 'react';

interface VINDetailsProps {
  vin: string;
  selectedFolders: string[];
}

const VINDetails: React.FC<VINDetailsProps> = ({ vin, selectedFolders }) => {
  const [history, setHistory] = useState(null);
  const [movement, setMovement] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Fetch both history and movement with same folder filter
        const [historyData, movementData] = await Promise.all([
          getVINHistory(vin, selectedFolders.length > 0 ? selectedFolders : undefined),
          getVINMovement(vin, selectedFolders.length > 0 ? selectedFolders : undefined)
        ]);
        
        setHistory(historyData);
        setMovement(movementData);
      } catch (error) {
        console.error('Error fetching VIN data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [vin, selectedFolders]);

  if (loading) return <div>Loading VIN details...</div>;

  return (
    <div className="vin-details">
      <h2>VIN: {vin}</h2>
      
      {history && (
        <div className="history-section">
          <h3>History ({history.total_spottings} spottings)</h3>
          <div className="history-list">
            {history.history.map((entry, index) => (
              <div key={index} className="history-item">
                <div className="folder">{entry.folder_name}</div>
                <div className="timestamp">{new Date(entry.spotted_at).toLocaleString()}</div>
                <div className="confidence">{entry.confidence}%</div>
                <div className="location">
                  {entry.latitude}, {entry.longitude}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {movement && (
        <div className="movement-section">
          <h3>Movement Path ({movement.total_points} points)</h3>
          <div className="movement-info">
            <div>Total Distance: {movement.total_distance_meters?.toFixed(2)} meters</div>
            <div>Folders: {movement.movement_points.map(p => p.folder_name).join(', ')}</div>
          </div>
          {/* Render movement path on map */}
          <div className="movement-map">
            {/* Your map component here */}
          </div>
        </div>
      )}
    </div>
  );
};

export default VINDetails;
```

### Usage Examples

```typescript
// 1. Get all data (no filtering)
const allHistory = await getVINHistory('1GNSK8KD3NR150647');
const allMovement = await getVINMovement('1GNSK8KD3NR150647');

// 2. Filter by single folder
const latestHistory = await getVINHistory('1GNSK8KD3NR150647', ['Run September 22 6:07 PM']);

// 3. Filter by multiple folders
const recentHistory = await getVINHistory('1GNSK8KD3NR150647', [
  'Run September 22 6:07 PM',
  'Run September 22 5:55 PM'
]);

// 4. Dashboard with filtering
const filteredDashboard = await getDashboard(['Run September 22 6:07 PM']);

// 5. VINs list with filtering
const filteredVINs = await getVINs(['Run September 22 6:07 PM']);
```

### URL Encoding Notes

When using folder names with spaces in URLs, make sure to URL encode them:

```typescript
// Correct - URL encoded
const folderName = encodeURIComponent('Run September 22 6:07 PM');
const url = `${API_BASE}/vins/123/history?folder_filter=${folderName}`;

// Or use URLSearchParams (recommended)
const params = new URLSearchParams();
params.append('folder_filter', 'Run September 22 6:07 PM');
const url = `${API_BASE}/vins/123/history?${params.toString()}`;
```

### CSS for Components

```css
.vin-details {
  padding: 1rem;
}

.history-section, .movement-section {
  margin-bottom: 2rem;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.history-item {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 2fr;
  gap: 1rem;
  padding: 0.5rem;
  border-bottom: 1px solid #eee;
}

.history-item:last-child {
  border-bottom: none;
}

.folder {
  font-weight: 500;
  color: #2563eb;
}

.timestamp {
  font-size: 0.875rem;
  color: #666;
}

.confidence {
  font-weight: 500;
  color: #059669;
}

.location {
  font-family: monospace;
  font-size: 0.875rem;
  color: #666;
}

.movement-info {
  display: flex;
  gap: 2rem;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  color: #666;
}

.movement-map {
  height: 400px;
  border: 1px solid #ddd;
  border-radius: 4px;
}
```

## Key Benefits

1. **Consistent Filtering**: All endpoints use the same folder filtering approach
2. **Performance**: Filter by specific folders to reduce data load
3. **User Experience**: Allow users to focus on specific drone runs
4. **Backward Compatible**: All endpoints work without filtering (existing behavior)
5. **Flexible**: Easy to implement "Select All" or "Select None" functionality

## Parameter Names

- **History & Movement**: Use `folder_filter` parameter
- **Dashboard & VINs**: Use `folders` parameter (existing)
- **All endpoints**: Accept comma-separated folder names

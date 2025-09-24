# AI VIN Description - Frontend Integration Guide

This guide explains how to integrate the new AI-powered VIN description feature into your frontend application.

## Overview

The QR Processing API now automatically generates detailed vehicle information using ChatGPT when creating QR codes. This includes:
- Make (Manufacturer)
- Model
- Year
- Body style
- Engine specification
- Assembly plant location

## API Changes

### Updated Response Format

The `/generate-qr` endpoint now returns an additional `ai_description` field:

```json
{
  "success": true,
  "qr_code": {
    "id": "uuid-string",
    "vin": "1HGCM82633A004352",
    "description": "User-provided description",
    "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 2003\n- Body: 4-door sedan\n- Engine: 3.0L V6\n- Assembly Plant: Marysville, OH, USA",
    "s3_url": "s3://bucket/qr_codes/...",
    "image_url": "https://signed-url...",
    "created_at": "2024-01-15T10:30:00",
    "size": 20
  },
  "message": "QR code generated successfully for VIN: 1HGCM82633A004352"
}
```

### Updated List Response

The `/qr-codes` endpoint now includes `ai_description` for each QR code:

```json
{
  "qr_codes": [
    {
      "id": "uuid-string",
      "vin": "1HGCM82633A004352",
      "description": "User description",
      "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 2003\n- Body: 4-door sedan\n- Engine: 3.0L V6\n- Assembly Plant: Marysville, OH, USA",
      "s3_url": "s3://bucket/qr_codes/...",
      "image_url": "https://signed-url...",
      "created_at": "2024-01-15T10:30:00",
      "size": 20
    }
  ],
  "total": 1
}
```

## Frontend Integration

### 1. Update TypeScript Interfaces

```typescript
interface QRCodeRecord {
  id: string;
  vin: string;
  description?: string;
  ai_description?: string;  // New field
  s3_url: string;
  image_url: string;
  created_at: string;
  size: number;
}

interface QRGenerateResponse {
  success: boolean;
  qr_code: QRCodeRecord;
  message: string;
}

interface QRListResponse {
  qr_codes: QRCodeRecord[];
  total: number;
}
```

### 2. Display AI Description in UI

#### Option A: Expandable Section
```tsx
import React, { useState } from 'react';

interface QRCodeCardProps {
  qrCode: QRCodeRecord;
}

const QRCodeCard: React.FC<QRCodeCardProps> = ({ qrCode }) => {
  const [showAIDescription, setShowAIDescription] = useState(false);

  const formatAIDescription = (aiDescription?: string) => {
    if (!aiDescription) return null;
    
    return aiDescription.split('\n').map((line, index) => (
      <div key={index} className="text-sm text-gray-600">
        {line}
      </div>
    ));
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold">{qrCode.vin}</h3>
          {qrCode.description && (
            <p className="text-gray-600">{qrCode.description}</p>
          )}
        </div>
        <img 
          src={qrCode.image_url} 
          alt={`QR Code for ${qrCode.vin}`}
          className="w-24 h-24"
        />
      </div>
      
      {qrCode.ai_description && (
        <div className="mt-4">
          <button
            onClick={() => setShowAIDescription(!showAIDescription)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            {showAIDescription ? 'Hide' : 'Show'} Vehicle Details
          </button>
          
          {showAIDescription && (
            <div className="mt-2 p-3 bg-gray-50 rounded-md">
              <h4 className="font-medium text-gray-800 mb-2">Vehicle Information:</h4>
              {formatAIDescription(qrCode.ai_description)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

#### Option B: Always Visible
```tsx
const QRCodeCard: React.FC<QRCodeCardProps> = ({ qrCode }) => {
  const formatAIDescription = (aiDescription?: string) => {
    if (!aiDescription) return null;
    
    return aiDescription.split('\n').map((line, index) => (
      <div key={index} className="flex">
        <span className="font-medium w-24 text-gray-700">
          {line.split(':')[0]}:
        </span>
        <span className="text-gray-600">
          {line.split(':').slice(1).join(':').trim()}
        </span>
      </div>
    ));
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-lg font-semibold mb-2">{qrCode.vin}</h3>
          {qrCode.description && (
            <p className="text-gray-600 mb-4">{qrCode.description}</p>
          )}
          <img 
            src={qrCode.image_url} 
            alt={`QR Code for ${qrCode.vin}`}
            className="w-32 h-32"
          />
        </div>
        
        {qrCode.ai_description && (
          <div className="bg-gray-50 p-4 rounded-md">
            <h4 className="font-medium text-gray-800 mb-3">Vehicle Details:</h4>
            {formatAIDescription(qrCode.ai_description)}
          </div>
        )}
      </div>
    </div>
  );
};
```

### 3. Enhanced QR Code Generation Form

```tsx
import React, { useState } from 'react';

const QRCodeGenerator: React.FC = () => {
  const [vin, setVin] = useState('');
  const [description, setDescription] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<QRGenerateResponse | null>(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('http://13.48.174.160:8002/generate-qr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          vin,
          description: description || undefined,
          size: 20
        }),
      });
      
      const data: QRGenerateResponse = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error generating QR code:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Generate QR Code with AI Description</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            VIN Number
          </label>
          <input
            type="text"
            value={vin}
            onChange={(e) => setVin(e.target.value)}
            placeholder="Enter VIN number (e.g., 1HGCM82633A004352)"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            maxLength={17}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Description (Optional)
          </label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <button
          onClick={handleGenerate}
          disabled={!vin || isGenerating}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
        >
          {isGenerating ? 'Generating...' : 'Generate QR Code with AI Description'}
        </button>
      </div>
      
      {result && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
          <h3 className="font-medium text-green-800 mb-2">QR Code Generated Successfully!</h3>
          <div className="space-y-2">
            <p><strong>VIN:</strong> {result.qr_code.vin}</p>
            {result.qr_code.description && (
              <p><strong>Description:</strong> {result.qr_code.description}</p>
            )}
            {result.qr_code.ai_description && (
              <div>
                <p className="font-medium text-green-800">AI-Generated Vehicle Details:</p>
                <div className="mt-1 text-sm text-gray-700">
                  {result.qr_code.ai_description.split('\n').map((line, index) => (
                    <div key={index}>{line}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
```

### 4. Enhanced QR Code List

```tsx
const QRCodeList: React.FC = () => {
  const [qrCodes, setQrCodes] = useState<QRCodeRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchQRCodes();
  }, []);

  const fetchQRCodes = async () => {
    try {
      const response = await fetch('http://13.48.174.160:8002/qr-codes');
      const data: QRListResponse = await response.json();
      setQrCodes(data.qr_codes);
    } catch (error) {
      console.error('Error fetching QR codes:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {qrCodes.map((qrCode) => (
        <QRCodeCard key={qrCode.id} qrCode={qrCode} />
      ))}
    </div>
  );
};
```

## Environment Setup

### Required Environment Variables

Add the following to your server environment:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Database Migration

Run the database migration to add the `ai_description` column:

```sql
ALTER TABLE qr_codes ADD COLUMN ai_description TEXT;
```

## Features

- **Automatic AI Description**: Generated for every QR code creation
- **Structured Format**: Consistent vehicle information format
- **Fallback Handling**: Graceful handling when AI service is unavailable
- **Caching**: AI descriptions are stored in the database
- **Fresh URLs**: Signed URLs are regenerated for each request

## Error Handling

The API gracefully handles AI service failures:
- If OpenAI is unavailable, `ai_description` will be `null`
- QR code generation continues normally
- No impact on core functionality

## Cost Considerations

- OpenAI API calls are made only during QR code generation
- Descriptions are cached in the database
- No additional API calls for existing QR codes
- Estimated cost: ~$0.001-0.002 per QR code generation

## Testing

Test the AI integration:

```bash
curl -X POST "http://13.48.174.160:8002/generate-qr" \
  -H "Content-Type: application/json" \
  -d '{
    "vin": "1HGCM82633A004352",
    "description": "Test VIN with AI",
    "size": 20
  }'
```

The response should include the `ai_description` field with detailed vehicle information.

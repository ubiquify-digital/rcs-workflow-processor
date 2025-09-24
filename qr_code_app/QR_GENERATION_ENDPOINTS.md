# QR Generation Endpoints

This document describes the new QR code generation endpoints added to the QR Processing API.

## Overview

The QR Processing API now includes endpoints for generating QR codes for VIN numbers, storing them in S3, and managing them in the database.

## Endpoints

### 1. Generate QR Code

**POST** `/generate-qr`

Generates a QR code for a VIN number and stores it in S3.

#### Request Body

```json
{
    "vin": "1HGBH41JXMN109186",
    "description": "Optional description",
    "size": 10
}
```

#### Parameters

- `vin` (string, required): VIN number to encode in QR code
- `description` (string, optional): Optional description for the QR code
- `size` (integer, optional): QR code size (1-20, default: 20 for ultra-high resolution A4 printing)

#### Response

```json
{
    "success": true,
    "qr_code": {
        "id": "uuid-string",
        "vin": "1HGBH41JXMN109186",
        "description": "Optional description",
        "s3_url": "s3://bucket/qr_codes/VIN_timestamp.png",
        "image_url": "https://signed-url-for-printing",
        "created_at": "2024-01-15T10:30:00",
        "size": 10
    },
    "message": "QR code generated successfully for VIN: 1HGBH41JXMN109186"
}
```

### 2. List QR Codes

**GET** `/qr-codes`

Lists all generated QR codes with fresh signed URLs.

#### Query Parameters

- `limit` (integer, optional): Number of records to return (default: 100, max: 1000)
- `offset` (integer, optional): Number of records to skip (default: 0)

#### Response

```json
{
    "qr_codes": [
        {
            "id": "uuid-string",
            "vin": "1HGBH41JXMN109186",
            "description": "Optional description",
            "s3_url": "s3://bucket/qr_codes/VIN_timestamp.png",
            "image_url": "https://signed-url-for-printing",
            "created_at": "2024-01-15T10:30:00",
            "size": 10
        }
    ],
    "total": 1
}
```

### 3. Get Specific QR Code

**GET** `/qr-codes/{qr_id}`

Gets a specific QR code by ID with fresh signed URL.

#### Response

```json
{
    "id": "uuid-string",
    "vin": "1HGBH41JXMN109186",
    "description": "Optional description",
    "s3_url": "s3://bucket/qr_codes/VIN_timestamp.png",
    "image_url": "https://signed-url-for-printing",
    "created_at": "2024-01-15T10:30:00",
    "size": 10
}
```

### 4. Delete QR Code

**DELETE** `/qr-codes/{qr_id}`

Deletes a QR code and its S3 object.

#### Response

```json
{
    "success": true,
    "message": "QR code uuid-string deleted successfully"
}
```

## Database Schema

The QR codes are stored in a `qr_codes` table with the following schema:

```sql
CREATE TABLE qr_codes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vin VARCHAR(17) NOT NULL,
    description TEXT,
    s3_url TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Usage Examples

### Generate a QR Code

```bash
curl -X POST "http://localhost:8002/generate-qr" \
  -H "Content-Type: application/json" \
  -d '{
    "vin": "1HGBH41JXMN109186",
    "description": "Test VIN QR Code",
    "size": 10
  }'
```

### List All QR Codes

```bash
curl "http://localhost:8002/qr-codes?limit=50&offset=0"
```

### Get Specific QR Code

```bash
curl "http://localhost:8002/qr-codes/{qr_id}"
```

### Delete QR Code

```bash
curl -X DELETE "http://localhost:8002/qr-codes/{qr_id}"
```

## Features

- **Ultra-High-Resolution A4 Printing**: QR codes optimized for professional A4 paper printing at 300 DPI
- **S3 Storage**: QR codes are automatically uploaded to S3 in the `qr_codes/` folder
- **Signed URLs**: Fresh signed URLs are generated for each request to prevent expiration
- **Database Caching**: All QR codes are stored in the database for quick retrieval
- **VIN Validation**: VIN numbers are sanitized for safe S3 key generation
- **Timestamped Files**: Each QR code gets a unique timestamp to prevent conflicts
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Print Quality**: Minimum 60px box size for ultra-crisp A4 printing, with minimal borders to maximize QR code size

## Environment Variables

The following environment variables are used:

- `S3_BUCKET`: S3 bucket name (default: "rcsstoragebucket")
- `SUPABASE_URL`: Supabase database URL
- `SUPABASE_KEY`: Supabase API key

## Testing

Use the provided test script to verify the endpoints:

```bash
python test_qr_generation.py
```

This will test all QR generation endpoints and verify they work correctly.

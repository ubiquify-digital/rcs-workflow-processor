# Reassignable QR Codes System

## Overview

The QR code system has been updated to support reassignable QR codes using a QR_CODE -> ID -> VIN mapping. This allows QR codes to be generated with unique IDs that can later be assigned to any VIN number, making them reusable and flexible.

## Architecture

### Before (Direct VIN Encoding)
```
QR Code Content = VIN Number
```

### After (ID-Based System)
```
QR Code Content = Unique ID -> Database Lookup -> VIN Number
```

## Database Schema Changes

### Migration Required

Run the migration script to update the existing `qr_codes` table:

```sql
-- Run migrate_qr_codes_to_reassignable.sql
```

### New Columns Added

- `qr_code_id`: The unique ID encoded in the QR code (VARCHAR(50), UNIQUE)
- `assigned_at`: Timestamp when VIN was assigned to this QR code
- `is_active`: Boolean flag for soft deletion (default: TRUE)

### Modified Columns

- `vin`: Now nullable (can be NULL initially, assigned later)

## API Changes

### 1. Generate QR Code (Updated)

**POST** `/generate-qr`

Now generates QR codes with unique IDs instead of direct VIN encoding.

#### Request Body
```json
{
    "description": "Optional description",
    "size": 10
}
```

#### Response
```json
{
    "success": true,
    "qr_code": {
        "id": "uuid-string",
        "qr_code_id": "ABC12345",
        "vin": null,
        "description": "Optional description",
        "ai_description": null,
        "s3_url": "s3://bucket/qr_codes/ABC12345_timestamp.png",
        "image_url": "https://signed-url-for-printing",
        "created_at": "2024-01-15T10:30:00",
        "assigned_at": null,
        "size": 10,
        "is_active": true
    },
    "message": "QR code generated successfully with ID: ABC12345. Use /reassign-qr to assign a VIN."
}
```

### 2. Reassign QR Code (New)

**POST** `/reassign-qr`

Assigns or reassigns a VIN to an existing QR code.

#### Request Body
```json
{
    "qr_code_id": "ABC12345",
    "vin": "1HGBH41JXMN109186"
}
```

#### Response
```json
{
    "success": true,
    "qr_code": {
        "id": "uuid-string",
        "qr_code_id": "ABC12345",
        "vin": "1HGBH41JXMN109186",
        "description": "Optional description",
        "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 2003\n- Body: 4-door sedan\n- Engine: 3.0L V6\n- Assembly Plant: Marysville, OH, USA",
        "s3_url": "s3://bucket/qr_codes/ABC12345_timestamp.png",
        "image_url": "https://signed-url-for-printing",
        "created_at": "2024-01-15T10:30:00",
        "assigned_at": "2024-01-15T11:30:00",
        "size": 10,
        "is_active": true
    },
    "message": "QR code ABC12345 successfully assigned to VIN 1HGBH41JXMN109186"
}
```

### 3. List QR Codes (Updated)

**GET** `/qr-codes`

Now includes the new fields in the response.

#### Response
```json
{
    "qr_codes": [
        {
            "id": "uuid-string",
            "qr_code_id": "ABC12345",
            "vin": "1HGBH41JXMN109186",
            "description": "Optional description",
            "ai_description": "- Make: Honda\n- Model: Accord\n- Year: 2003\n- Body: 4-door sedan\n- Engine: 3.0L V6\n- Assembly Plant: Marysville, OH, USA",
            "s3_url": "s3://bucket/qr_codes/ABC12345_timestamp.png",
            "image_url": "https://signed-url-for-printing",
            "created_at": "2024-01-15T10:30:00",
            "assigned_at": "2024-01-15T11:30:00",
            "size": 10,
            "is_active": true
        }
    ],
    "total": 1
}
```

## QR Code Processing

### Backward Compatibility

The system maintains backward compatibility:

1. **Direct VIN QR Codes**: If a QR code contains a 17-character alphanumeric string, it's treated as a direct VIN
2. **ID-Based QR Codes**: If a QR code contains an 8-character ID, it's looked up in the database to resolve to a VIN

### Resolution Logic

```python
def resolve_qr_code_to_vin(qr_content: str) -> Optional[str]:
    # Check if content is already a VIN (17 characters, alphanumeric)
    if len(qr_content) == 17 and qr_content.isalnum():
        return qr_content
    
    # Otherwise, treat as QR code ID and look up VIN
    # Database lookup: qr_codes table where qr_code_id = content
    return database_lookup_result
```

## Usage Workflow

### 1. Generate QR Code
```bash
curl -X POST "http://localhost:8002/generate-qr" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Vehicle QR Code",
    "size": 10
  }'
```

### 2. Assign VIN to QR Code
```bash
curl -X POST "http://localhost:8002/reassign-qr" \
  -H "Content-Type: application/json" \
  -d '{
    "qr_code_id": "ABC12345",
    "vin": "1HGBH41JXMN109186"
  }'
```

### 3. Reassign to Different VIN
```bash
curl -X POST "http://localhost:8002/reassign-qr" \
  -H "Content-Type: application/json" \
  -d '{
    "qr_code_id": "ABC12345",
    "vin": "1HGBH41JXMN109187"
  }'
```

## Benefits

1. **Reusability**: QR codes can be reassigned to different vehicles
2. **Flexibility**: Generate QR codes in advance and assign VINs later
3. **Backward Compatibility**: Existing VIN-based QR codes continue to work
4. **Traceability**: Track when VINs were assigned to QR codes
5. **Soft Deletion**: QR codes can be deactivated without losing data

## Database Functions

### New Functions Added

- `get_qr_record_by_qr_code_id(qr_code_id)`: Get QR code by ID
- `assign_vin_to_qr_code(qr_code_id, vin)`: Assign VIN to QR code
- `generate_unique_qr_code_id()`: Generate unique 8-character ID
- `resolve_qr_code_to_vin(qr_content)`: Resolve QR content to VIN

### Updated Functions

- `store_qr_record()`: Now stores QR codes without VIN initially
- `create_s3_key_for_qr()`: Uses QR code ID instead of VIN
- QR processing logic: Now resolves IDs to VINs during processing

## Migration Notes

1. **Existing QR Codes**: Will continue to work as direct VIN QR codes
2. **Database Migration**: Run the migration script to add new columns
3. **API Compatibility**: All existing endpoints continue to work
4. **New Features**: Use new endpoints for reassignable functionality

## Error Handling

- **QR Code Not Found**: Returns 404 when trying to reassign non-existent QR code
- **Invalid VIN**: VIN validation still applies
- **Database Errors**: Proper error handling for database operations
- **Backward Compatibility**: Graceful handling of both old and new QR code formats

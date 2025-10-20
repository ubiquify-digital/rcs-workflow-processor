-- Migration script to make QR codes reassignable
-- This modifies the existing qr_codes table to support QR_CODE -> ID -> VIN mapping

-- Add new columns to support reassignable QR codes (without unique constraint initially)
ALTER TABLE qr_codes 
ADD COLUMN IF NOT EXISTS qr_code_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS assigned_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Update existing records to have qr_code_id = vin (for backward compatibility)
-- Handle duplicates by adding a suffix to make them unique
WITH vin_counts AS (
    SELECT vin, COUNT(*) as count, MIN(id) as first_id
    FROM qr_codes 
    GROUP BY vin
),
duplicate_vins AS (
    SELECT vin FROM vin_counts WHERE count > 1
)
UPDATE qr_codes 
SET 
    qr_code_id = CASE 
        WHEN qr_codes.vin IN (SELECT vin FROM duplicate_vins) 
        THEN qr_codes.vin || '_' || EXTRACT(EPOCH FROM qr_codes.created_at)::INTEGER
        ELSE qr_codes.vin 
    END,
    assigned_at = created_at, 
    is_active = TRUE
WHERE qr_code_id IS NULL;

-- Now add the unique constraint after all records have been updated
ALTER TABLE qr_codes ADD CONSTRAINT qr_codes_qr_code_id_unique UNIQUE (qr_code_id);

-- Create index for the new qr_code_id column
CREATE INDEX IF NOT EXISTS idx_qr_codes_qr_code_id ON qr_codes(qr_code_id);
CREATE INDEX IF NOT EXISTS idx_qr_codes_is_active ON qr_codes(is_active);

-- Make vin nullable since QR codes can exist without VIN assignment initially
ALTER TABLE qr_codes ALTER COLUMN vin DROP NOT NULL;

-- Add function to generate unique QR code IDs
CREATE OR REPLACE FUNCTION generate_qr_code_id()
RETURNS VARCHAR(50) AS $$
DECLARE
    new_id VARCHAR(50);
    exists_count INTEGER;
BEGIN
    LOOP
        -- Generate a random 8-character alphanumeric ID
        new_id := upper(substring(md5(random()::text) from 1 for 8));
        
        -- Check if this ID already exists
        SELECT COUNT(*) INTO exists_count 
        FROM qr_codes 
        WHERE qr_code_id = new_id;
        
        -- If ID doesn't exist, we can use it
        IF exists_count = 0 THEN
            EXIT;
        END IF;
    END LOOP;
    
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- Update comments to reflect new functionality
COMMENT ON TABLE qr_codes IS 'Stores QR codes with reassignable VIN associations';
COMMENT ON COLUMN qr_codes.qr_code_id IS 'The ID encoded in the QR code (reassignable)';
COMMENT ON COLUMN qr_codes.vin IS 'Vehicle Identification Number (can be reassigned)';
COMMENT ON COLUMN qr_codes.assigned_at IS 'Timestamp when VIN was assigned to this QR code';
COMMENT ON COLUMN qr_codes.is_active IS 'Whether the QR code is active (for soft deletion)';

-- Add constraint to ensure qr_code_id is not null for new records
ALTER TABLE qr_codes ADD CONSTRAINT qr_codes_qr_code_id_not_null 
CHECK (qr_code_id IS NOT NULL);

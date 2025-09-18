-- QR Code Generator Database Schema
-- This schema stores generated QR codes for VIN numbers

-- Create qr_codes table
CREATE TABLE IF NOT EXISTS qr_codes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vin VARCHAR(17) NOT NULL,
    description TEXT,
    s3_url TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_qr_codes_vin ON qr_codes(vin);
CREATE INDEX IF NOT EXISTS idx_qr_codes_created_at ON qr_codes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qr_codes_s3_url ON qr_codes(s3_url);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_qr_codes_updated_at 
    BEFORE UPDATE ON qr_codes 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE qr_codes IS 'Stores generated QR codes for VIN numbers';
COMMENT ON COLUMN qr_codes.id IS 'Unique identifier for the QR code record';
COMMENT ON COLUMN qr_codes.vin IS 'Vehicle Identification Number encoded in the QR code';
COMMENT ON COLUMN qr_codes.description IS 'Optional description for the QR code';
COMMENT ON COLUMN qr_codes.s3_url IS 'S3 URL where the QR code image is stored';
COMMENT ON COLUMN qr_codes.size IS 'Size of the QR code (1-20)';
COMMENT ON COLUMN qr_codes.created_at IS 'Timestamp when the QR code was created';
COMMENT ON COLUMN qr_codes.updated_at IS 'Timestamp when the record was last updated';

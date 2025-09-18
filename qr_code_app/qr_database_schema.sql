-- QR Processing API Database Schema Extension
-- Add these tables to your existing Supabase database

-- Create table for QR processed images
CREATE TABLE IF NOT EXISTS qr_processed_images (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id UUID NOT NULL,
    filename TEXT NOT NULL,
    s3_input_url TEXT NOT NULL,
    image_signed_url TEXT,
    timestamp TIMESTAMPTZ,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    qr_results JSONB,
    processing_status TEXT DEFAULT 'success',
    error_message TEXT,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Add unique constraint to prevent duplicate processing
    UNIQUE(s3_input_url)
);

-- Create table for QR processed folders
CREATE TABLE IF NOT EXISTS qr_processed_folders (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id UUID UNIQUE NOT NULL,
    folder_name TEXT NOT NULL,
    s3_input_folder_url TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'queued',
    total_images INTEGER,
    processed_images INTEGER DEFAULT 0,
    successful_images INTEGER DEFAULT 0,
    failed_images INTEGER DEFAULT 0,
    total_qr_codes INTEGER DEFAULT 0,
    error_message TEXT,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for QR images
CREATE INDEX IF NOT EXISTS idx_qr_processed_images_task_id ON qr_processed_images(task_id);
CREATE INDEX IF NOT EXISTS idx_qr_processed_images_s3_url ON qr_processed_images(s3_input_url);
CREATE INDEX IF NOT EXISTS idx_qr_processed_images_timestamp ON qr_processed_images(timestamp);
CREATE INDEX IF NOT EXISTS idx_qr_processed_images_location ON qr_processed_images(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_qr_processed_images_status ON qr_processed_images(processing_status);

-- Create indexes for QR folders
CREATE INDEX IF NOT EXISTS idx_qr_processed_folders_task_id ON qr_processed_folders(task_id);
CREATE INDEX IF NOT EXISTS idx_qr_processed_folders_s3_url ON qr_processed_folders(s3_input_folder_url);
CREATE INDEX IF NOT EXISTS idx_qr_processed_folders_status ON qr_processed_folders(status);
CREATE INDEX IF NOT EXISTS idx_qr_processed_folders_folder_name ON qr_processed_folders(folder_name);
CREATE INDEX IF NOT EXISTS idx_qr_processed_folders_created_at ON qr_processed_folders(created_at);

-- Add function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_qr_folder_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_qr_folder_updated_at_trigger
    BEFORE UPDATE ON qr_processed_folders
    FOR EACH ROW
    EXECUTE FUNCTION update_qr_folder_updated_at();

-- Create view for folder statistics
CREATE OR REPLACE VIEW qr_folder_stats AS
SELECT 
    f.id,
    f.task_id,
    f.folder_name,
    f.s3_input_folder_url,
    f.status,
    f.total_images,
    f.processed_images,
    f.successful_images,
    f.failed_images,
    f.total_qr_codes,
    f.processing_started_at,
    f.processing_completed_at,
    f.created_at,
    f.updated_at,
    CASE 
        WHEN f.total_images > 0 THEN 
            ROUND((f.processed_images::DECIMAL / f.total_images) * 100, 2)
        ELSE 0 
    END as completion_percentage,
    CASE 
        WHEN f.processing_completed_at IS NOT NULL AND f.processing_started_at IS NOT NULL THEN
            EXTRACT(EPOCH FROM (f.processing_completed_at - f.processing_started_at))
        ELSE NULL
    END as processing_duration_seconds
FROM qr_processed_folders f;

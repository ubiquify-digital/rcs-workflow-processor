-- Supabase database schema for image processing
-- Run this in your Supabase SQL editor

-- Create table for processed images
CREATE TABLE IF NOT EXISTS processed_images (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id UUID NOT NULL,
    filename TEXT NOT NULL,
    s3_input_url TEXT,
    s3_output_url TEXT,
    timestamp TIMESTAMPTZ,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    altitude DECIMAL(10, 2),
    detections JSONB,
    processing_status TEXT DEFAULT 'success',
    error_message TEXT,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_processed_images_task_id ON processed_images(task_id);
CREATE INDEX IF NOT EXISTS idx_processed_images_timestamp ON processed_images(timestamp);
CREATE INDEX IF NOT EXISTS idx_processed_images_location ON processed_images(latitude, longitude);

-- Create table for processed folders
CREATE TABLE IF NOT EXISTS processed_folders (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    task_id UUID UNIQUE NOT NULL,
    folder_name TEXT NOT NULL,
    s3_input_folder_url TEXT NOT NULL,
    s3_output_folder_url TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    total_images INTEGER,
    processed_images INTEGER DEFAULT 0,
    successful_images INTEGER DEFAULT 0,
    failed_images INTEGER DEFAULT 0,
    error_message TEXT,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for folders
CREATE INDEX IF NOT EXISTS idx_processed_folders_task_id ON processed_folders(task_id);
CREATE INDEX IF NOT EXISTS idx_processed_folders_status ON processed_folders(status);
CREATE INDEX IF NOT EXISTS idx_processed_folders_folder_name ON processed_folders(folder_name);

-- Add bounding box visualization URL field to qr_processed_images table
-- Run this migration to add support for storing bounding box visualizations

ALTER TABLE qr_processed_images 
ADD COLUMN IF NOT EXISTS bbox_visualization_url TEXT;

-- Add comment for documentation
COMMENT ON COLUMN qr_processed_images.bbox_visualization_url IS 'S3 URL for the bounding box visualization image showing detected QR codes';

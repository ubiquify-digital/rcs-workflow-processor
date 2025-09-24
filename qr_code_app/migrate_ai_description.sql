-- Migration to add ai_description column to qr_codes table
-- Run this migration to enable AI VIN description features

-- Add ai_description column to existing qr_codes table
ALTER TABLE qr_codes ADD COLUMN IF NOT EXISTS ai_description TEXT;

-- Add comment for documentation
COMMENT ON COLUMN qr_codes.ai_description IS 'AI-generated vehicle description from OpenAI ChatGPT';

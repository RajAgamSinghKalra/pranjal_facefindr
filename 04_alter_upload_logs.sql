-- 04_alter_upload_logs.sql

-- Add a new flexible column for storing query-specific metadata
-- This column will be JSONB, allowing you to store dictionaries, lists, etc.
ALTER TABLE upload_logs
ADD COLUMN IF NOT EXISTS query_metadata JSONB DEFAULT '{}';

-- Drop the existing CHECK constraint to add new allowed values
ALTER TABLE upload_logs
DROP CONSTRAINT IF EXISTS upload_logs_status_check;

-- Add the new CHECK constraint with updated status values
ALTER TABLE upload_logs
ADD CONSTRAINT upload_logs_status_check
CHECK (status IN (
    'processed_new_faces',      -- For successful uploads with new faces
    'processed_matched_faces',  -- For successful uploads matching existing faces
    'error',                    -- For general errors during processing/upload
    'skipped',                  -- For images skipped (e.g., no faces detected)
    'queried',                  -- For general image query
    'query_no_match',           -- For image query where no similar faces were found
    'query_error'               -- For errors specific to query processing
));

COMMENT ON COLUMN upload_logs.query_metadata IS 'Stores JSON metadata related to the log entry, e.g., matched face IDs and scores for queries.';
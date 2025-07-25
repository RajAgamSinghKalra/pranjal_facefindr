-- 01_setup_extensions_and_tables.sql

-- Ensure we are in the correct database if not already connected.
-- \c your_database_name; -- Uncomment and replace if you need to switch databases

-- 1. Enable the 'vector' extension for efficient vector similarity search.
-- This is crucial for storing and querying face embeddings.
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the 'clusters' table
-- This table represents unique individuals (or groups of similar faces).
-- Each cluster can eventually be assigned a human-readable name.
CREATE TABLE IF NOT EXISTS clusters (
    id SERIAL PRIMARY KEY,
    -- 'person_name' will store the identified name for a cluster.
    -- It can be NULL initially for unknown clusters, and later updated.
    -- UNIQUE constraint ensures no two clusters have the same name.
    person_name TEXT UNIQUE NULL,
    -- Optional: reference to a representative face for the cluster, useful for UI.
    -- This reference will be added as a foreign key after face_vectors table is created.
    representative_face_id INTEGER NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- 3. Create the 'face_vectors' table
-- This table stores details for each detected and extracted face.
CREATE TABLE IF NOT EXISTS face_vectors (
    id SERIAL PRIMARY KEY,
    -- Path to the original source image (group photo or individual photo).
    original_image_path TEXT NOT NULL,
    -- Path to the specifically extracted and saved face image file.
    -- This should be unique for each extracted face to prevent duplicates.
    extracted_face_path TEXT NOT NULL UNIQUE,
    -- Bounding box coordinates: [x1, y1, x2, y2].
    -- Using JSONB for flexibility, allowing for potential future variations
    -- like confidence scores or landmarks within the same field.
    -- Example: '[10, 20, 100, 120]' or '{"x1": 10, "y1": 20, "x2": 100, "y2": 120, "confidence": 0.98}'
    bounding_box JSONB NOT NULL,
    -- The 512-dimensional face embedding (feature vector).
    embedding VECTOR(512) NOT NULL,
    -- Foreign key to the 'clusters' table.
    -- This links a detected face to a recognized individual/cluster.
    -- It can be NULL initially if a face hasn't been clustered yet or is noise.
    -- ON DELETE SET NULL: If a cluster is deleted, faces previously linked to it
    -- will have their cluster_id set to NULL instead of being deleted,
    -- allowing for re-clustering or manual assignment.
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Add a foreign key constraint for representative_face_id in 'clusters'
-- This needs to be done after 'face_vectors' table is created.
-- DEFERRABLE INITIALLY DEFERRED: allows inserting into clusters first, then updating representative_face_id.
-- This is useful if you create a cluster, then later assign a face to be its representative.
ALTER TABLE clusters
ADD CONSTRAINT fk_representative_face
FOREIGN KEY (representative_face_id) REFERENCES face_vectors(id) DEFERRABLE INITIALLY DEFERRED;


-- 4. Create the 'upload_logs' table
-- This table tracks information about uploaded images and their processing status.
CREATE TABLE IF NOT EXISTS upload_logs (
    id SERIAL PRIMARY KEY,
    -- Path to the original image file that was uploaded.
    uploaded_image_path TEXT NOT NULL,
    -- Status of the upload processing.
    -- 'processed_new_faces': New faces were detected and added to face_vectors.
    -- 'processed_matched_faces': Faces were detected and matched to existing clusters.
    -- 'error': An error occurred during processing.
    -- 'skipped': Image was skipped for some reason (e.g., no faces found).
    status TEXT NOT NULL CHECK (status IN ('processed_new_faces', 'processed_matched_faces', 'error', 'skipped')),
    -- Optional: ID of a specific face_vector if the upload led to its creation/update.
    -- Can be NULL if no faces were processed or on error.
    associated_face_id INTEGER REFERENCES face_vectors(id) ON DELETE SET NULL,
    -- Optional: ID of a specific cluster if the upload led to its creation/update/match.
    -- Can be NULL if no clusters were affected or on error.
    associated_cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    -- A message providing more details about the log entry.
    message TEXT,
    logged_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Add comments for better database documentation
COMMENT ON TABLE clusters IS 'Stores information about identified individuals or groups of similar faces.';
COMMENT ON COLUMN clusters.person_name IS 'The human-readable name assigned to a cluster (e.g., "John Doe"). NULL if not yet identified.';
COMMENT ON COLUMN clusters.representative_face_id IS 'ID of a face_vector that best represents this cluster (e.g., for UI display).';

COMMENT ON TABLE face_vectors IS 'Stores details for each detected and extracted face, including its embedding and bounding box.';
COMMENT ON COLUMN face_vectors.original_image_path IS 'Path to the source image from which the face was extracted.';
COMMENT ON COLUMN face_vectors.extracted_face_path IS 'Path to the cropped and saved image of the detected face.';
COMMENT ON COLUMN face_vectors.bounding_box IS 'JSONB array or object containing the coordinates of the detected face [x1, y1, x2, y2].';
COMMENT ON COLUMN face_vectors.embedding IS 'The high-dimensional feature vector generated by FaceNet for the face.';
COMMENT ON COLUMN face_vectors.cluster_id IS 'Foreign key linking this face to a cluster (identified person). NULL if unclustered or noise.';

COMMENT ON TABLE upload_logs IS 'Logs the processing status of uploaded images.';
COMMENT ON COLUMN upload_logs.uploaded_image_path IS 'The path of the original image file that was uploaded.';
COMMENT ON COLUMN upload_logs.status IS 'The outcome of processing the uploaded image (e.g., new faces, matched faces, error).';
COMMENT ON COLUMN upload_logs.associated_face_id IS 'References a specific face_vector entry if relevant to the log.';
COMMENT ON COLUMN upload_logs.associated_cluster_id IS 'References a specific cluster entry if relevant to the log.';
COMMENT ON COLUMN upload_logs.message IS 'Detailed message about the log entry.';

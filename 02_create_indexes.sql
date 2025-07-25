-- 02_create_indexes.sql

-- Index for efficient vector similarity search on face embeddings.
-- IVFFLAT is suitable for cosine similarity searches.
-- The 'lists' parameter (e.g., 100) should be tuned based on your dataset size.
-- A general rule is 'num_rows / 1000' or 'sqrt(num_rows)' but start with a reasonable value.
-- For a smaller dataset (thousands), 100-500 lists might be good.
-- Make sure to ANALYZE the table after significant data inserts to update statistics.
CREATE INDEX IF NOT EXISTS idx_face_vectors_embedding
ON face_vectors USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); -- Adjust 'lists' based on your data size and performance needs

-- Index on cluster_id for fast lookups of faces belonging to a specific cluster.
CREATE INDEX IF NOT EXISTS idx_face_vectors_cluster_id
ON face_vectors (cluster_id);

-- Index on original_image_path for quick retrieval of faces from a specific source image.
CREATE INDEX IF NOT EXISTS idx_face_vectors_original_image_path
ON face_vectors (original_image_path);

-- Index on person_name in clusters for quick lookup by name.
CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_person_name_unique
ON clusters (person_name)
WHERE person_name IS NOT NULL; -- Index only non-NULL names for efficiency

-- Index on uploaded_image_path in upload_logs for fast log retrieval.
CREATE INDEX IF NOT EXISTS idx_upload_logs_uploaded_image_path
ON upload_logs (uploaded_image_path);

-- Index on creation timestamps if you frequently query by time ranges.
CREATE INDEX IF NOT EXISTS idx_face_vectors_created_at
ON face_vectors (created_at);

CREATE INDEX IF NOT EXISTS idx_clusters_created_at
ON clusters (created_at);

CREATE INDEX IF NOT EXISTS idx_upload_logs_logged_at
ON upload_logs (logged_at);

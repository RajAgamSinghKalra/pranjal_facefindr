-- 03_create_functions_and_triggers.sql

-- Function to automatically update the 'updated_at' column to the current timestamp.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for 'clusters' table: Updates 'updated_at' on each row update.
DROP TRIGGER IF EXISTS trg_clusters_updated_at ON clusters;
CREATE TRIGGER trg_clusters_updated_at
BEFORE UPDATE ON clusters
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Trigger for 'face_vectors' table: Updates 'updated_at' on each row update.
DROP TRIGGER IF EXISTS trg_face_vectors_updated_at ON face_vectors;
CREATE TRIGGER trg_face_vectors_updated_at
BEFORE UPDATE ON face_vectors
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

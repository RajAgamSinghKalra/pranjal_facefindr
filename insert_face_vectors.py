import os
import json
import psycopg2

# CONFIGURATION
DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}
JSON_DIR = 'group_faces'  # Directory containing your .json files

def log_upload_event(cur, uploaded_image_path, status, message=None, associated_face_id=None, associated_cluster_id=None, query_metadata=None):
    sql = """
        INSERT INTO upload_logs (
            uploaded_image_path, status, message, associated_face_id, associated_cluster_id, query_metadata
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """
    cur.execute(sql, (
        uploaded_image_path,
        status,
        message,
        associated_face_id,
        associated_cluster_id,
        json.dumps(query_metadata) if query_metadata else None
    ))

def insert_face(face, cur):
    # Map JSON keys to DB columns
    original_image_path = face.get('original_image')
    extracted_face_path = face.get('image_path')  # If you have a separate path for cropped face, use that
    bounding_box = json.dumps(face.get('bounding_box'))
    embedding = face.get('embedding')
    landmarks = json.dumps(face.get('landmarks')) if 'landmarks' in face else None

    sql = """
        INSERT INTO face_vectors (
            original_image_path,
            extracted_face_path,
            bounding_box,
            embedding
        ) VALUES (%s, %s, %s, %s)
        ON CONFLICT (extracted_face_path) DO NOTHING
        RETURNING id
    """
    cur.execute(sql, (original_image_path, extracted_face_path, bounding_box, embedding))
    face_id = cur.fetchone()
    # Log the upload event
    log_upload_event(
        cur,
        uploaded_image_path=original_image_path,
        status='processed_new_faces',
        message=f"Inserted face from {extracted_face_path}",
        associated_face_id=face_id[0] if face_id else None,
        query_metadata={"bounding_box": face.get('bounding_box'), "landmarks": face.get('landmarks')}
    )

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    count = 0
    for filename in os.listdir(JSON_DIR):
        if filename.endswith('.json'):
            json_path = os.path.join(JSON_DIR, filename)
            print(f"Processing {json_path}")
            with open(json_path, 'r') as f:
                face = json.load(f)
            insert_face(face, cur)
            count += 1
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {count} faces into the database.")

if __name__ == '__main__':
    main()

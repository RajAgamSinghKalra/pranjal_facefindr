from group_photo_processor import GroupPhotoProcessor
import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from shutil import copyfile

# --- CONFIGURATION ---
DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}
QUERY_IMAGE_PATH = '/Users/pranjalsharma/Desktop/face recognition copy/facefindr/static/uploads/query_97fa811b-8c8b-42ae-9f22-efb71e2d6c83.jpg'

QUERY_CROP_DIR = "query_crops"

# Make sure the directory exists
os.makedirs(QUERY_CROP_DIR, exist_ok=True)

def fetch_all_db_embeddings():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT extracted_face_path, embedding FROM face_vectors")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    paths = [row[0] for row in rows]
    def parse_embedding(embedding):
        if isinstance(embedding, str):
            return [float(x) for x in embedding.strip('[]').split(',')]
        return embedding
    embeddings = np.array([parse_embedding(row[1]) for row in rows])
    return paths, embeddings

def main():
    processor = GroupPhotoProcessor()
    print("Detecting and cropping faces in query image using shared logic...")
    extracted_faces = processor.detect_and_crop_faces(QUERY_IMAGE_PATH)
    if not extracted_faces:
        print("No faces found in query image.")
        return

    results = []
    for face_data in extracted_faces:
        face_img = face_data['image']
        # Save crop to QUERY_CROP_DIR
        import uuid
        crop_filename = f"query_face_{uuid.uuid4().hex[:8]}.jpg"
        crop_path = os.path.join(QUERY_CROP_DIR, crop_filename)
        face_img.save(crop_path, 'JPEG', quality=95, optimize=True, progressive=True)
        # Generate embedding using the same logic as the database
        embedding = processor.extract_face_embedding(crop_path)
        if embedding is not None:
            results.append((crop_path, embedding))
            print(f"Cropped face saved at: {crop_path}")
            # Optionally, display the crop using PIL
            from PIL import Image
            img = Image.open(crop_path)
            img.show()
        else:
            print(f"Failed to generate embedding for: {crop_path}")

    db_paths, db_embeddings = fetch_all_db_embeddings()
    for crop_path, query_embedding in results:
        print(f"\nComparing for crop: {crop_path}")
        sims = cosine_similarity([query_embedding], db_embeddings)[0]
        print("Similarity scores with all database faces:")
        for db_path, sim in zip(db_paths, sims):
            print(f"  {db_path}: {sim:.4f}")
        top_n = np.argsort(sims)[::-1][:5]
        print("\nTop 5 most similar faces in DB:")
        for idx in top_n:
            print(f"  {db_paths[idx]} (similarity: {sims[idx]:.4f})")
        print(f"Best match: {db_paths[top_n[0]]} (similarity: {sims[top_n[0]]:.4f})")

if __name__ == '__main__':
    main()
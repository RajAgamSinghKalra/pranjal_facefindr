from group_photo_processor import GroupPhotoProcessor
import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- CONFIGURATION ---
DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}
ROOT_DIR = Path(__file__).resolve().parent
QUERY_IMAGE_PATH = str(ROOT_DIR / 'facefindr' / 'static' / 'uploads' / 'query_97fa811b-8c8b-42ae-9f22-efb71e2d6c83.jpg')

def fetch_db_embedding(face_path):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM face_vectors WHERE extracted_face_path = %s", (face_path,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        print(f"No embedding found in DB for {face_path}")
        return None
    embedding = row[0]
    if isinstance(embedding, str):
        embedding = [float(x) for x in embedding.strip('[]').split(',')]
    return np.array(embedding, dtype=np.float32)

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
    print("Cropping and embedding query image...")
    results = processor.crop_and_embed_query_image(QUERY_IMAGE_PATH)
    if not results:
        print("No faces found in query image.")
        return

    db_paths, db_embeddings = fetch_all_db_embeddings()
    for crop_path, query_embedding in results:
        print(f"\nComparing for crop: {crop_path}")
        print("Query embedding (first 10):", query_embedding[:10])
        sims = cosine_similarity([query_embedding], db_embeddings)[0]
        top_idx = np.argmax(sims)
        print(f"Most similar DB face: {db_paths[top_idx]}")
        print(f"Cosine similarity: {sims[top_idx]:.6f}")
        # Optionally, print top N matches
        top_n = np.argsort(sims)[::-1][:5]
        for idx in top_n:
            print(f"  {db_paths[idx]} (similarity: {sims[idx]:.4f})")

if __name__ == '__main__':
    main()
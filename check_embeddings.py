import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

def fetch_embeddings():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id, extracted_face_path, embedding FROM face_vectors")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids = [row[0] for row in rows]
    paths = [row[1] for row in rows]
    def parse_embedding(embedding):
        if isinstance(embedding, str):
            # Remove brackets and split by comma
            return [float(x) for x in embedding.strip('[]').split(',')]
        return embedding
    embeddings = np.array([parse_embedding(row[2]) for row in rows])
    return embeddings, paths

def main():
    embeddings, paths = fetch_embeddings()
    n = len(embeddings)
    print(f"Loaded {n} embeddings from database.")

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    print("Cosine similarity matrix (rounded to 3 decimals):")
    print(np.round(sim_matrix, 3))

    # Check for identical embeddings
    identical_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(embeddings[i], embeddings[j], atol=1e-6):
                identical_pairs.append((paths[i], paths[j]))

    if identical_pairs:
        print("\nIdentical embeddings found:")
        for a, b in identical_pairs:
            print(f"{a} and {b}")
    else:
        print("\nAll embeddings are unique (no identical pairs found).")

    # Optionally, check for very similar embeddings (cosine similarity > 0.99, but not exactly 1)
    similar_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if 0.99 < sim_matrix[i, j] < 1.0:
                similar_pairs.append((paths[i], paths[j], sim_matrix[i, j]))
    if similar_pairs:
        print("\nHighly similar (but not identical) embeddings found (cosine similarity > 0.99):")
        for a, b, sim in similar_pairs:
            print(f"{a} and {b} (similarity: {sim:.4f})")
    else:
        print("\nNo highly similar (cosine similarity > 0.99) non-identical embeddings found.")

if __name__ == '__main__':
    main()
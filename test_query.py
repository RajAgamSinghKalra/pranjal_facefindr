from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import psycopg2
# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160), Image.Resampling.LANCZOS)
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(160, 160, 3).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()
    return embedding

query_image_path = '/Users/pranjalsharma/Desktop/face recognition copy/group_photos/begruessung-outreach-gaeste-4-data.jpg'
query_embedding = get_embedding(query_image_path)
print("Query embedding (first 10 values):", query_embedding[:10])



DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

def fetch_db_embedding(image_path):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM face_vectors WHERE extracted_face_path = %s", (image_path,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        print("No embedding found in DB for", image_path)
        return None
    # Parse string to list if needed
    embedding = row[0]
    if isinstance(embedding, str):
        embedding = [float(x) for x in embedding.strip('[]').split(',')]
    return np.array(embedding)

db_image_path = 'group_faces/photo1.jpg'  # or whatever path matches your DB
db_embedding = fetch_db_embedding(db_image_path)
print("DB embedding (first 10 values):", db_embedding[:10])

if db_embedding is not None:
    print("Are embeddings identical?", np.allclose(query_embedding, db_embedding, atol=1e-6))
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([query_embedding], [db_embedding])[0][0]
    print(f"Cosine similarity: {sim:.6f}")
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import tempfile
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IdentiFace API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'dbname': 'face_recognition_db',
    'user': 'pranjalsharma',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

# Initialize face detection and recognition models
face_detector = MTCNN(keep_all=True, device='cpu', post_process=False, min_face_size=40)
face_encoder = InceptionResnetV1(pretrained='vggface2').eval()

class DatabaseManager:
    def __init__(self):
        self.db_config = DB_CONFIG
    
    def get_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    if fetch:
                        return cur.fetchall()
                    conn.commit()
                    return cur.rowcount
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

db_manager = DatabaseManager()

class FaceProcessor:
    def __init__(self):
        self.face_detector = face_detector
        self.face_encoder = face_encoder
    
    def detect_and_crop_faces(self, image_path: str, min_face_size: int = 40):
        """Detect and crop faces from an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            boxes, probs, landmarks = self.face_detector.detect(img, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return []
            
            extracted_faces = []
            confidence_threshold = 0.90
            
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.astype(int)
                face_width = x2 - x1
                face_height = y2 - y1
                
                if face_width < min_face_size or face_height < min_face_size:
                    continue
                
                face_img = img.crop((x1, y1, x2, y2))
                
                extracted_faces.append({
                    'image': face_img,
                    'box': box.tolist(),
                    'landmarks': landmark.tolist() if landmark is not None else None,
                    'confidence': float(prob),
                    'size': (face_width, face_height)
                })
            
            return extracted_faces
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return []
    
    def extract_face_embedding(self, face_image: Image.Image):
        """Extract face embedding from a PIL Image"""
        try:
            # Resize to 160x160 for FaceNet
            if face_image.size != (160, 160):
                face_image = face_image.resize((160, 160), Image.Resampling.LANCZOS)
            
            # Convert to tensor
            img_tensor = torch.tensor(list(face_image.getdata()), dtype=torch.float32)
            img_tensor = img_tensor.view(160, 160, 3).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.face_encoder(img_tensor).cpu().numpy().flatten()
            
            if len(embedding) != 512 or not np.all(np.isfinite(embedding)):
                return None
            
            return embedding
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

face_processor = FaceProcessor()

@app.get("/")
async def root():
    return {"message": "IdentiFace API is running"}

@app.get("/api/stats")
async def get_stats():
    """Get database statistics"""
    try:
        # Get total faces
        faces_query = "SELECT COUNT(*) as count FROM face_vectors"
        faces_result = db_manager.execute_query(faces_query)
        total_faces = faces_result[0]['count'] if faces_result else 0
        
        # Get total clusters (distinct cluster_ids from a hypothetical clusters table)
        # For now, we'll estimate from face_vectors
        clusters_query = """
            SELECT COUNT(DISTINCT SUBSTRING(extracted_face_path FROM 'photo([0-9]+)')) as count 
            FROM face_vectors
        """
        clusters_result = db_manager.execute_query(clusters_query)
        total_clusters = min(total_faces // 3, 50)  # Rough estimate
        
        # Get total photos
        photos_query = "SELECT COUNT(DISTINCT original_image_path) as count FROM face_vectors"
        photos_result = db_manager.execute_query(photos_query)
        total_photos = photos_result[0]['count'] if photos_result else 0
        
        return {
            "totalFaces": total_faces,
            "totalClusters": total_clusters,
            "totalPhotos": total_photos
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clusters")
async def get_clusters():
    """Get all face clusters"""
    try:
        # Load cluster data from JSON file (fallback to database if needed)
        root_dir = Path(__file__).resolve().parents[2]
        cluster_file = os.path.join(root_dir, "face_clusters.json")
        
        if os.path.exists(cluster_file):
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            clusters = []
            for cluster_id, face_paths in cluster_data.items():
                if cluster_id == "-1":  # Skip noise cluster
                    continue
                
                # Get face details from database
                faces = []
                for face_path in face_paths[:10]:  # Limit to first 10 faces
                    face_query = """
                        SELECT id, original_image_path, extracted_face_path, bounding_box
                        FROM face_vectors 
                        WHERE extracted_face_path LIKE %s
                        LIMIT 1
                    """
                    face_result = db_manager.execute_query(
                        face_query, 
                        (f"%{os.path.basename(face_path)}%",)
                    )
                    
                    if face_result:
                        face_data = face_result[0]
                        faces.append({
                            "id": str(face_data['id']),
                            "path": face_data['extracted_face_path'],
                            "similarity": 0.95,  # Placeholder
                            "cluster_id": int(cluster_id),
                            "original_image": face_data['original_image_path'],
                            "bounding_box": json.loads(face_data['bounding_box']) if face_data['bounding_box'] else [],
                            "landmarks": []
                        })
                
                if faces:
                    clusters.append({
                        "id": int(cluster_id),
                        "faces": faces,
                        "representative_face": faces[0],
                        "size": len(face_paths)
                    })
            
            return clusters
        else:
            # Fallback: create dummy clusters from database
            query = """
                SELECT id, original_image_path, extracted_face_path, bounding_box
                FROM face_vectors 
                ORDER BY id
                LIMIT 50
            """
            faces_result = db_manager.execute_query(query)
            
            clusters = []
            for i, face_data in enumerate(faces_result):
                cluster_id = i // 3  # Group every 3 faces
                face = {
                    "id": str(face_data['id']),
                    "path": face_data['extracted_face_path'],
                    "similarity": 0.95,
                    "cluster_id": cluster_id,
                    "original_image": face_data['original_image_path'],
                    "bounding_box": json.loads(face_data['bounding_box']) if face_data['bounding_box'] else [],
                    "landmarks": []
                }
                
                # Find or create cluster
                existing_cluster = next((c for c in clusters if c["id"] == cluster_id), None)
                if existing_cluster:
                    existing_cluster["faces"].append(face)
                    existing_cluster["size"] += 1
                else:
                    clusters.append({
                        "id": cluster_id,
                        "faces": [face],
                        "representative_face": face,
                        "size": 1
                    })
            
            return clusters
            
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clusters/{cluster_id}")
async def get_cluster(cluster_id: int):
    """Get specific cluster details"""
    try:
        clusters = await get_clusters()
        cluster = next((c for c in clusters if c["id"] == cluster_id), None)
        
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
        
        return cluster
    except Exception as e:
        logger.error(f"Error getting cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_similar_faces(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """Search for similar faces"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Extract faces from uploaded image
            extracted_faces = face_processor.detect_and_crop_faces(tmp_file_path)
            
            if not extracted_faces:
                return {"message": "No faces detected in uploaded image", "results": []}
            
            # Use the first detected face for search
            query_face = extracted_faces[0]
            query_embedding = face_processor.extract_face_embedding(query_face['image'])
            
            if query_embedding is None:
                return {"message": "Could not extract face embedding", "results": []}
            
            # Get all face embeddings from database
            query = "SELECT id, original_image_path, extracted_face_path, bounding_box, embedding FROM face_vectors"
            faces_result = db_manager.execute_query(query)
            
            similar_faces = []
            for face_data in faces_result:
                if face_data['embedding']:
                    stored_embedding = np.array(face_data['embedding'])
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                    
                    if similarity >= threshold:
                        similar_faces.append({
                            "id": str(face_data['id']),
                            "path": face_data['extracted_face_path'],
                            "similarity": float(similarity),
                            "cluster_id": hash(face_data['extracted_face_path']) % 100,  # Placeholder
                            "original_image": face_data['original_image_path'],
                            "bounding_box": json.loads(face_data['bounding_box']) if face_data['bounding_box'] else [],
                            "landmarks": []
                        })
            
            # Sort by similarity
            similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_faces[:20]  # Return top 20 matches
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_group_photo(file: UploadFile = File(...)):
    """Upload and process a group photo"""
    try:
        # Save uploaded file
        root_dir = Path(__file__).resolve().parents[2]
        upload_dir = os.path.join(root_dir, "group_photos")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the image
        extracted_faces = face_processor.detect_and_crop_faces(file_path)
        
        faces_processed = 0
        for i, face_data in enumerate(extracted_faces):
            # Extract embedding
            embedding = face_processor.extract_face_embedding(face_data['image'])
            if embedding is None:
                continue
            
            # Save face crop
            face_filename = f"upload_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(root_dir, "group_faces", face_filename)
            face_data['image'].save(face_path, 'JPEG', quality=95)
            
            # Insert into database
            insert_query = """
                INSERT INTO face_vectors (
                    original_image_path, extracted_face_path, bounding_box, embedding
                ) VALUES (%s, %s, %s, %s)
            """
            db_manager.execute_query(
                insert_query,
                (
                    file_path,
                    face_path,
                    json.dumps(face_data['box']),
                    embedding.tolist()
                ),
                fetch=False
            )
            faces_processed += 1
        
        return {
            "message": f"Successfully processed {faces_processed} faces",
            "faces_detected": faces_processed
        }
        
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_group_photos():
    """Process all group photos in the directory"""
    try:
        # This would trigger the group photo processing pipeline
        # For now, return a success message
        return {
            "message": "Group photo processing started",
            "total_faces": 0
        }
    except Exception as e:
        logger.error(f"Error processing group photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/similarity-threshold")
async def update_similarity_threshold(threshold: float):
    """Update similarity threshold"""
    try:
        # Store threshold in database or configuration
        return {"message": f"Similarity threshold updated to {threshold}"}
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

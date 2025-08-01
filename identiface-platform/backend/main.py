from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import json
import logging
import uuid
from datetime import datetime
import random
import os
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IdentiFace API",
    description="AI-Powered Face Recognition and Clustering Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 🎯 Paths to cropped faces and group photos
# Use environment variables if provided, otherwise default to directories
# located at the project root.
ROOT_DIR = Path(__file__).resolve().parents[2]
FACE_IMAGES_DIR = Path(os.getenv("FACE_IMAGES_DIR", ROOT_DIR / "group_faces"))
GROUP_PHOTOS_DIR = Path(os.getenv("GROUP_PHOTOS_DIR", ROOT_DIR / "group_photos"))

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files for serving images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get all available face images from your cropped faces directory
def get_available_face_images():
    """Get all cropped face images from the directory"""
    if not FACE_IMAGES_DIR.exists():
        logger.error(f"❌ Cropped faces directory not found: {FACE_IMAGES_DIR}")
        return []
    
    # Get all image files (including different cases)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
    face_images = []
    
    logger.info(f"🔍 Scanning cropped faces directory: {FACE_IMAGES_DIR}")
    
    for ext in image_extensions:
        pattern = str(FACE_IMAGES_DIR / ext)
        found_files = glob.glob(pattern)
        if found_files:
            logger.info(f"Found {len(found_files)} files matching {ext}")
        face_images.extend(found_files)
    
    logger.info(f"✅ Total cropped faces found: {len(face_images)}")
    
    # Log first few files for debugging
    if face_images:
        logger.info("📸 Sample cropped face images:")
        for i, img in enumerate(face_images[:5]):
            file_size = os.path.getsize(img) if os.path.exists(img) else 0
            logger.info(f"  {i+1}. {os.path.basename(img)} ({file_size:,} bytes)")
    else:
        logger.warning("⚠️  No cropped face images found!")
        # List what's actually in the directory
        try:
            all_files = list(FACE_IMAGES_DIR.iterdir())
            logger.info(f"Directory contains {len(all_files)} total files:")
            for f in all_files[:10]:  # Show first 10 files
                logger.info(f"  - {f.name}")
        except Exception as e:
            logger.error(f"Error reading directory: {e}")
    
    return face_images

# Get all available group photos
def get_available_group_photos():
    """Get all group photos from the directory"""
    if not GROUP_PHOTOS_DIR.exists():
        logger.warning(f"Group photos directory not found: {GROUP_PHOTOS_DIR}")
        GROUP_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        return []
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
    group_photos = []
    
    for ext in image_extensions:
        found_files = glob.glob(str(GROUP_PHOTOS_DIR / ext))
        group_photos.extend(found_files)
    
    logger.info(f"Found {len(group_photos)} group photos in {GROUP_PHOTOS_DIR}")
    return group_photos

# Mock data generator with real cropped face images
class MockDataGenerator:
    def __init__(self):
        self.face_images = get_available_face_images()
        self.group_photos = get_available_group_photos()
        logger.info(f"MockDataGenerator initialized with {len(self.face_images)} cropped faces and {len(self.group_photos)} group photos")
        
        # If no images found, log warning
        if not self.face_images:
            logger.warning("No cropped face images found! Check the directory path.")
        
        if not self.group_photos:
            logger.warning("No group photos found! Using placeholder data.")
            self.group_photos = [f"placeholder_group_{i}.jpg" for i in range(10)]
    
    def get_face_image_by_index(self, index: int):
        """Get a cropped face image by index to ensure consistency"""
        if not self.face_images:
            return None
        return self.face_images[index % len(self.face_images)]
    
    def get_group_photo_by_index(self, index: int):
        """Get a group photo by index to ensure consistency"""
        if not self.group_photos:
            return f"group_photo_{index}.jpg"
        return self.group_photos[index % len(self.group_photos)]
    
    def generate_face(self, cluster_id: int, face_idx: int) -> Dict[str, Any]:
        # Use cluster_id to get consistent cropped face image
        face_image_path = self.get_face_image_by_index(cluster_id + face_idx)
        face_id = f"face_{cluster_id}_{face_idx}_{uuid.uuid4().hex[:8]}"
        
        if face_image_path and os.path.exists(face_image_path):
            face_filename = os.path.basename(face_image_path)
            face_url = f"http://localhost:8000/api/face-image/{face_filename}"
        else:
            # Fallback placeholder
            face_filename = f"placeholder_{cluster_id}_{face_idx}.jpg"
            face_url = f"http://localhost:8000/api/face-image/{face_filename}"
        
        # Use cluster_id to get consistent group photo
        group_photo_path = self.get_group_photo_by_index(cluster_id)
        group_photo_name = os.path.basename(group_photo_path) if isinstance(group_photo_path, str) else f"group_{cluster_id}.jpg"
        
        return {
            "id": face_id,
            "path": face_url,
            "similarity": round(0.95 - face_idx * 0.01 - random.uniform(0, 0.05), 3),
            "cluster_id": cluster_id,
            "original_image": group_photo_name,
            "original_image_path": group_photo_path if isinstance(group_photo_path, str) else None,
            "bounding_box": [
                100 + face_idx * 10 + random.randint(-5, 5),
                100 + face_idx * 5 + random.randint(-5, 5),
                200 + face_idx * 10 + random.randint(-5, 5),
                200 + face_idx * 5 + random.randint(-5, 5)
            ],
            "landmarks": []
        }
    
    def generate_cluster(self, cluster_id: int) -> Dict[str, Any]:
        size = random.randint(3, 8)
        faces = [
            self.generate_face(cluster_id, face_idx)
            for face_idx in range(size)
        ]
        
        return {
            "id": cluster_id,
            "faces": faces,
            "representative_face": faces[0] if faces else None,
            "size": size
        }

# Global state for demo
app_state = {
    "total_photos": 0,
    "total_faces": 0,
    "total_clusters": 0,
    "clusters": [],
    "uploaded_images": {},
    "data_generator": None,
    "last_updated": datetime.now()
}

def initialize_mock_data():
    """Initialize mock data on startup"""
    app_state["data_generator"] = MockDataGenerator()
    
    # Get actual counts from your directories
    face_images = get_available_face_images()
    group_photos = get_available_group_photos()
    
    # Generate clusters based on available cropped faces
    clusters = []
    num_clusters = min(15, len(face_images) // 3) if face_images else 5
    
    for cluster_id in range(1, num_clusters + 1):
        cluster = app_state["data_generator"].generate_cluster(cluster_id)
        clusters.append(cluster)
    
    total_faces_in_clusters = sum(cluster["size"] for cluster in clusters)
    
    app_state.update({
        "clusters": clusters,
        "total_faces": len(face_images),
        "total_clusters": len(clusters),
        "total_photos": len(group_photos),
        "last_updated": datetime.now()
    })
    
    logger.info(f"Initialized {len(clusters)} clusters with {len(face_images)} total cropped faces")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting IdentiFace API v2.0.0")
    logger.info(f"Cropped faces directory: {FACE_IMAGES_DIR}")
    logger.info(f"Group photos directory: {GROUP_PHOTOS_DIR}")
    initialize_mock_data()
    logger.info("✅ Mock data initialized successfully")

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "IdentiFace API v2.0.0 is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    face_images = get_available_face_images()
    group_photos = get_available_group_photos()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "database": "file_system",
        "available_face_images": len(face_images),
        "available_group_photos": len(group_photos),
        "face_images_dir": str(FACE_IMAGES_DIR),
        "group_photos_dir": str(GROUP_PHOTOS_DIR),
        "face_images_exist": FACE_IMAGES_DIR.exists(),
        "group_photos_exist": GROUP_PHOTOS_DIR.exists(),
        "features": {
            "face_detection": "enabled",
            "clustering": "enabled",
            "search": "enabled",
            "upload": "enabled"
        }
    }

@app.get("/api/face-image/{filename}")
async def get_face_image(filename: str):
    """Serve cropped face images from the faces directory"""
    try:
        # Clean the filename to prevent directory traversal
        clean_filename = os.path.basename(filename)
        file_path = FACE_IMAGES_DIR / clean_filename
        
        logger.info(f"Serving cropped face image: {file_path}")
        
        if not file_path.exists():
            logger.warning(f"Cropped face image not found: {file_path}")
            # Try to find a similar file (case insensitive)
            for existing_file in FACE_IMAGES_DIR.glob("*"):
                if existing_file.name.lower() == clean_filename.lower():
                    logger.info(f"Found case-insensitive match: {existing_file}")
                    return FileResponse(
                        existing_file,
                        media_type="image/jpeg",
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "Access-Control-Allow-Origin": "*"
                        }
                    )
            
            # Return 404 if no file found
            raise HTTPException(status_code=404, detail=f"Cropped face image not found: {clean_filename}")
        
        return FileResponse(
            file_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving cropped face image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/original-photo/{filename}")
async def get_original_photo(filename: str):
    """Serve original group photos"""
    try:
        # Clean the filename to prevent directory traversal
        clean_filename = os.path.basename(filename)
        file_path = GROUP_PHOTOS_DIR / clean_filename
        
        logger.info(f"Serving original photo: {file_path}")
        
        if not file_path.exists():
            logger.warning(f"Original photo not found: {file_path}")
            # Try to find a similar file (case insensitive)
            for existing_file in GROUP_PHOTOS_DIR.glob("*"):
                if existing_file.name.lower() == clean_filename.lower():
                    logger.info(f"Found case-insensitive match: {existing_file}")
                    return FileResponse(
                        existing_file,
                        media_type="image/jpeg",
                        headers={
                            "Cache-Control": "public, max-age=3600",
                            "Access-Control-Allow-Origin": "*"
                        }
                    )
            
            raise HTTPException(status_code=404, detail=f"Original photo not found: {clean_filename}")
        
        return FileResponse(
            file_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving original photo {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/images")
async def debug_images():
    """Debug endpoint to check available images"""
    face_images = get_available_face_images()
    group_photos = get_available_group_photos()
    
    return {
        "face_images_dir": str(FACE_IMAGES_DIR),
        "group_photos_dir": str(GROUP_PHOTOS_DIR),
        "face_images_exist": FACE_IMAGES_DIR.exists(),
        "group_photos_exist": GROUP_PHOTOS_DIR.exists(),
        "face_images_count": len(face_images),
        "group_photos_count": len(group_photos),
        "face_images_sample": [os.path.basename(f) for f in face_images[:10]],
        "group_photos_sample": [os.path.basename(f) for f in group_photos[:10]],
        "face_images_full": face_images[:5],  # First 5 full paths
        "group_photos_full": group_photos[:5],  # First 5 full paths
        "current_working_directory": os.getcwd(),
        "sample_face_url": f"http://localhost:8000/api/face-image/{os.path.basename(face_images[0])}" if face_images else "No images found",
        "sample_group_url": f"http://localhost:8000/api/original-photo/{os.path.basename(group_photos[0])}" if group_photos else "No images found"
    }

@app.get("/api/stats")
async def get_stats():
    """Get application statistics"""
    try:
        face_images = get_available_face_images()
        group_photos = get_available_group_photos()
        
        return {
            "totalFaces": len(face_images),
            "totalClusters": app_state["total_clusters"],
            "totalPhotos": len(group_photos),
            "lastUpdated": app_state["last_updated"].isoformat(),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clusters")
async def get_clusters():
    """Get all face clusters using cropped face images"""
    try:
        clusters = app_state["clusters"]
        logger.info(f"Returning {len(clusters)} clusters")
        
        # Log first cluster for debugging
        if clusters:
            first_cluster = clusters[0]
            logger.info(f"First cluster: ID={first_cluster['id']}, faces={len(first_cluster['faces'])}")
            if first_cluster['faces']:
                first_face = first_cluster['faces'][0]
                logger.info(f"First face path: {first_face['path']}")
        
        return clusters
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clusters/{cluster_id}")
async def get_cluster(cluster_id: int):
    """Get specific cluster details"""
    try:
        cluster = next(
            (c for c in app_state["clusters"] if c["id"] == cluster_id),
            None
        )
        
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
        
        return cluster
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_similar_faces(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """🎯 MAIN SEARCH FUNCTION - Returns your cropped face images as search results"""
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Save the uploaded file (for future processing if needed)
        file_id = uuid.uuid4().hex
        file_path = UPLOAD_DIR / f"search_{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Search initiated with threshold {threshold}")
        
        # 🎯 GET ALL YOUR CROPPED FACE IMAGES
        face_images = get_available_face_images()
        group_photos = get_available_group_photos()
        
        if not face_images:
            logger.warning("No cropped face images available for search")
            return []
        
        logger.info(f"Found {len(face_images)} cropped faces to search through")
        
        # 🎯 RETURN CROPPED FACES AS SEARCH RESULTS
        results = []
        max_results = 20  # Limit results for performance
        
        # Shuffle the faces to get different results each time
        shuffled_faces = random.sample(face_images, min(len(face_images), max_results * 2))
        
        for i, face_image_path in enumerate(shuffled_faces):
            if len(results) >= max_results:
                break
                
            # Generate realistic similarity score
            similarity = random.uniform(0.65, 0.98)
            
            # Only include faces that meet the threshold
            if similarity >= threshold:
                face_filename = os.path.basename(face_image_path)
                cluster_id = random.randint(1, app_state["total_clusters"])
                
                # Get a corresponding group photo (if available)
                group_photo_path = group_photos[i % len(group_photos)] if group_photos else None
                group_photo_name = os.path.basename(group_photo_path) if group_photo_path else f"group_photo_{i}.jpg"
                
                results.append({
                    "id": f"cropped_face_{file_id}_{i}",
                    "path": f"http://localhost:8000/api/face-image/{face_filename}",  # 🎯 URL to your cropped face
                    "similarity": round(similarity, 3),
                    "cluster_id": cluster_id,
                    "original_image": group_photo_name,
                    "original_image_path": group_photo_path,
                    "bounding_box": [
                        80 + i * 15 + random.randint(-10, 10),
                        85 + i * 10 + random.randint(-10, 10),
                        180 + i * 15 + random.randint(-10, 10),
                        185 + i * 10 + random.randint(-10, 10)
                    ],
                    "landmarks": []
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"✅ Search completed: {len(results)} cropped faces returned (threshold: {threshold})")
        
        # Log sample results for debugging
        if results:
            logger.info("Sample search results:")
            for i, result in enumerate(results[:3]):
                logger.info(f"  {i+1}. {result['path']} (similarity: {result['similarity']})")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_group_photo(file: UploadFile = File(...)):
    """Upload and process a group photo"""
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Save the uploaded file
        file_id = uuid.uuid4().hex
        file_path = UPLOAD_DIR / f"upload_{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store the uploaded image info
        app_state["uploaded_images"][file_id] = {
            "filename": file.filename,
            "path": str(file_path),
            "upload_time": datetime.now().isoformat()
        }
        
        # Simulate processing
        faces_detected = random.randint(1, 6)
        
        # Update stats
        app_state["total_photos"] += 1
        app_state["total_faces"] += faces_detected
        app_state["last_updated"] = datetime.now()
        
        logger.info(f"Photo uploaded: {file.filename}, detected {faces_detected} faces")
        
        return {
            "message": f"Successfully processed {file.filename}",
            "faces_detected": faces_detected,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "file_id": file_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_group_photos():
    """Process all group photos in the directory"""
    try:
        # Simulate batch processing
        total_faces = random.randint(20, 100)
        new_clusters = random.randint(2, 8)
        
        # Update stats
        app_state["total_faces"] += total_faces
        app_state["total_clusters"] += new_clusters
        app_state["last_updated"] = datetime.now()
        
        logger.info(f"Batch processing completed: {total_faces} faces, {new_clusters} new clusters")
        
        return {
            "message": "Group photo processing completed successfully",
            "total_faces": total_faces,
            "new_clusters": new_clusters,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing group photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/similarity-threshold")
async def update_similarity_threshold(threshold: float):
    """Update similarity threshold"""
    try:
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        
        logger.info(f"Similarity threshold updated to {threshold}")
        
        return {
            "message": f"Similarity threshold updated to {threshold}",
            "threshold": threshold,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list-faces")
async def list_all_faces():
    """List all available cropped face images with their URLs"""
    try:
        face_images = get_available_face_images()
        
        faces_list = []
        for i, img_path in enumerate(face_images):
            filename = os.path.basename(img_path)
            file_size = os.path.getsize(img_path) if os.path.exists(img_path) else 0
            
            faces_list.append({
                "index": i,
                "filename": filename,
                "full_path": img_path,
                "url": f"http://localhost:8000/api/face-image/{filename}",
                "size_bytes": file_size,
                "exists": os.path.exists(img_path)
            })
        
        return {
            "total_faces": len(faces_list),
            "directory": str(FACE_IMAGES_DIR),
            "faces": faces_list[:50]  # Limit to first 50 for performance
        }
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import faiss
from tqdm import tqdm
import pickle
from collections import defaultdict
import cv2
from sklearn.metrics.pairwise import cosine_distances

class GroupPhotoProcessor:
    def __init__(self, output_dir="group_faces", similarity_threshold=0.8,
                 distance_threshold=0.7):
        """
        Initialize group photo processor
        
        Args:
            output_dir: Directory to save extracted faces from group photos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.distance_threshold = distance_threshold
        
        # Initialize models
        # Keep MTCNN's internal min_face_size at a reasonable level, maybe slightly lower than default 20
        # or closer to 40, as very small internal min_face_size can increase false positives significantly.
        # Let's try 40 here.
        self.face_detector = MTCNN(keep_all=True, device='cpu', post_process=False, min_face_size=40) 
        self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval()

        # Faiss index for fast similarity search
        self.index = None
        
        # Store extracted faces and their embeddings
        self.extracted_faces = []
        self.face_embeddings = []
        self.face_paths = []
        self.global_face_counter = 1  # Counter for naming faces as photo1, photo2, ...
        
    def detect_and_crop_faces(self, image_path, min_face_size=40):
        """
        Helper to load image, detect faces/landmarks, apply all filtering, and crop faces.
        Returns a list of dicts with face crop and metadata.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            boxes, probs, landmarks = self.face_detector.detect(img, landmarks=True)
            if boxes is None or len(boxes) == 0:
                print(f"No faces detected in {image_path}")
                return []
            extracted_faces = []
            confidence_threshold = 0.90
            eye_distance_threshold = 25
            landmark_margin = 0.0  # Loosened to allow faces with landmarks near the edge
            blur_threshold = 100.0
            min_face_area = 1600
            aspect_ratio_range = (0.6, 1.6)
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < confidence_threshold:
                    print(f"Face {i} from {image_path} below confidence threshold ({prob:.2f}), skipping")
                    continue
                x1, y1, x2, y2 = box.astype(int)
                face_width = x2 - x1
                face_height = y2 - y1
                if face_width < min_face_size or face_height < min_face_size:
                    print(f"Face {i} from {image_path} too small ({face_width}x{face_height}), skipping")
                    continue
                if face_width * face_height < min_face_area:
                    print(f"Face {i} from {image_path} area too small ({face_width*face_height}), skipping")
                    continue
                if face_width == 0:
                    print(f"Face {i} from {image_path} has zero width, skipping")
                    continue
                aspect_ratio = face_height / face_width
                if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]):
                    print(f"Face {i} from {image_path} has unfavorable aspect ratio ({aspect_ratio:.2f}), skipping")
                    continue
                if landmark is None:
                    print(f"Face {i} from {image_path} has missing landmarks, skipping")
                    continue
                if landmark.shape != (5, 2):
                    print(f"Face {i} from {image_path} has invalid landmark shape ({landmark.shape}), skipping")
                    continue
                left_eye, right_eye = landmark[0], landmark[1]
                eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
                if eye_distance < eye_distance_threshold:
                    print(f"Face {i} from {image_path} eye distance too small ({eye_distance:.1f}), skipping")
                    continue
                margin_x = face_width * landmark_margin
                margin_y = face_height * landmark_margin
                all_inside = True
                for (lx, ly) in landmark:
                    if not (x1 + margin_x < lx < x2 - margin_x and y1 + margin_y < ly < y2 - margin_y):
                        all_inside = False
                        break
                if not all_inside:
                    print(f"Face {i} from {image_path} has landmarks too close to edge, skipping")
                    continue
                face_img = img.crop((x1, y1, x2, y2))
                face_img_cv = np.array(face_img)
                if face_img_cv.ndim == 3 and face_img_cv.shape[2] == 3:
                    face_img_gray = cv2.cvtColor(face_img_cv, cv2.COLOR_RGB2GRAY)
                elif face_img_cv.ndim == 2:
                    face_img_gray = face_img_cv
                else:
                    print(f"Face {i} from {image_path} has unsupported image format for blur check, skipping blur check.")
                    lap_var = blur_threshold + 1
                if 'lap_var' not in locals() or lap_var <= blur_threshold:
                    if face_img_gray.size == 0:
                        print(f"Face {i} from {image_path} has empty grayscale image, skipping.")
                        continue
                    lap_var = cv2.Laplacian(face_img_gray, cv2.CV_64F).var()
                    if lap_var < blur_threshold:
                        print(f"Face {i} from {image_path} is too blurry (Laplacian variance {lap_var:.1f}), skipping")
                        continue
                extracted_faces.append({
                    'image': face_img,
                    'box': box,
                    'landmarks': landmark,
                    'index': i,
                    'size': (face_width, face_height)
                })
            print(f"Extracted {len(extracted_faces)} faces from {image_path}")
            return extracted_faces
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return []

    def extract_faces_from_image(self, image_path, min_face_size=40):
        """
        Extract all faces from a group photo (uses shared helper)
        """
        return self.detect_and_crop_faces(image_path, min_face_size)
    
    def save_extracted_faces(self, extracted_faces, base_filename):
        """Save extracted faces to disk with global sequential naming"""
        saved_paths = []
        face_index_to_path = {}
        for i, face_data in enumerate(extracted_faces):
            face_img = face_data['image']
            # Align face if landmarks are available for better embedding quality
            if face_data.get('landmarks') is not None:
                aligned = self.align_face(face_img, face_data['landmarks'])
                if aligned is not None:
                    face_img = aligned
            # Save the crop (aligned if possible)
            filename = f"photo{self.global_face_counter}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            face_img.save(filepath, 'JPEG', quality=95, optimize=True, progressive=True)
            saved_paths.append(filepath)
            face_index_to_path[i] = filepath
            self.extracted_faces.append({
                'path': filepath,
                'box': face_data['box'],
                'bounding_box': [int(x) for x in face_data['box']],
                'landmarks': face_data.get('landmarks', None),
                'size': face_data['size'],
                'original_image': base_filename
            })
            self.global_face_counter += 1
        return saved_paths
    
    def extract_face_embedding(self, image_path, landmarks=None):
        """
        Extract face embedding from a single image, with robust error handling and device/model management.
        """
        try:
            # Device/model caching
            if not hasattr(self, '_device'):
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)
            # Validate image
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None
            img = Image.open(image_path).convert('RGB')
            # Ensure consistent size
            if img.size != (160, 160):
                img = img.resize((160, 160), Image.Resampling.LANCZOS)
            # Convert to tensor and normalize
            img_array = np.array(img, dtype=np.float32)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # match training preprocessing
            img_tensor = img_tensor.unsqueeze(0).to(self._device)
            with torch.no_grad():
                embedding = self.face_encoder(img_tensor).cpu().numpy().flatten()
            # Ensure unit length to make cosine similarity meaningful
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            if len(embedding) != 512 or not np.all(np.isfinite(embedding)):
                print(f"Invalid embedding for {image_path}")
                return None
            return embedding
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def align_face(self, img, landmarks):
        """Align face image using landmarks (eyes, nose, mouth)"""
        try:
            # Reference points for alignment (from MTCNN paper, 160x160)
            ref_landmarks = np.array([
                [38.2946, 51.6963],   # left eye
                [122.5318, 51.5014],  # right eye
                [80.0,    92.3655],    # nose
                [54.0, 133.0356],     # left mouth
                [108.0, 132.2330]     # right mouth
            ], dtype=np.float32)
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Compute similarity transform
            # Check if landmarks are valid (e.g., not all NaN or inf) - already done before calling this function
            
            # For robustness, try both default and LMEDS if initial fails.
            # LMEDS is more robust to outliers but can be slower.
            # Default method is RANSAC.
            tfm = None
            try:
                tfm = cv2.estimateAffinePartial2D(landmarks, ref_landmarks, method=cv2.RANSAC)[0]
            except cv2.error as e:
                # If RANSAC fails, try LMEDS
                print(f"RANSAC failed, trying LMEDS for alignment: {e}")
                tfm = cv2.estimateAffinePartial2D(landmarks, ref_landmarks, method=cv2.LMEDS)[0]

            if tfm is not None:
                aligned_img = cv2.warpAffine(np.array(img), tfm, (160, 160), borderValue=0.0)
                return Image.fromarray(aligned_img)
            else:
                return None
        except Exception as e:
            print(f"Error aligning face: {str(e)}")
            return None
    
    def process_group_photos(self, input_dir, min_face_size=40): # Adjusted default min_face_size
        """
        Process all group photos in a directory
        
        Args:
            input_dir: Directory containing group photos
            min_face_size: Minimum face size to extract
        """
        print(f"Processing group photos from {input_dir}...")
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        total_faces_extracted = 0
        
        for img_path in tqdm(image_files, desc="Processing group photos"):
            # Extract faces from group photo
            extracted_faces = self.extract_faces_from_image(img_path, min_face_size)
            
            if extracted_faces:
                # Save extracted faces
                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                saved_paths = self.save_extracted_faces(extracted_faces, base_filename)
                total_faces_extracted += len(saved_paths)
        
        print(f"Total faces extracted: {total_faces_extracted}")
        return total_faces_extracted
    
    def cluster_extracted_faces(self, min_cluster_size=2, min_samples=1):
        """
        Cluster the extracted faces to group similar people using HDBSCAN
        Args:
            min_cluster_size: HDBSCAN min_cluster_size parameter
            min_samples: HDBSCAN min_samples parameter
        """
        import json
        import hdbscan
        print("Extracting embeddings from all extracted faces...")
        embeddings = []
        face_paths = []
        temp_extracted_faces_for_clustering = [] 
        for face_data in tqdm(self.extracted_faces, desc="Extracting embeddings"):
            embedding = self.extract_face_embedding(face_data['path'], landmarks=face_data.get('landmarks'))
            if embedding is not None:
                embeddings.append(embedding)
                face_paths.append(face_data['path']) 
                temp_extracted_faces_for_clustering.append(face_data)
                # Save per-face JSON file
                landmarks = face_data.get('landmarks', None)
                if isinstance(landmarks, np.ndarray):
                    landmarks = landmarks.tolist()
                json_data = {
                    'image_path': face_data['path'],  # Cropped face image path
                    'original_image': face_data.get('original_image', None),  # Original group photo path
                    'embedding': embedding.tolist(),
                    'bounding_box': [int(x) for x in face_data['box']],
                    'landmarks': landmarks
                }
                json_path = os.path.splitext(face_data['path'])[0] + '.json'
                with open(json_path, 'w') as jf:
                    json.dump(json_data, jf, indent=2)
        if not embeddings:
            print("No valid embeddings extracted")
            return None
        self.face_embeddings = np.array(embeddings)
        self.face_paths = face_paths
        print(f"Extracted {len(embeddings)} embeddings")
        print("Clustering faces with HDBSCAN...")
        distance_matrix = cosine_distances(self.face_embeddings).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        cluster_labels = clusterer.fit_predict(distance_matrix)
        from collections import defaultdict
        cluster_groups = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            original_face_data = temp_extracted_faces_for_clustering[i]
            cluster_groups[label].append({
                'path': self.face_paths[i],
                'embedding': self.face_embeddings[i],
                'original_data': original_face_data,
                'bounding_box': original_face_data.get('bounding_box', None),
                'landmarks': original_face_data.get('landmarks', None)
            })
        n_clusters = len(cluster_groups) - (1 if -1 in cluster_groups else 0)
        print(f"\nClustering Results:")
        print(f"Number of clusters: {n_clusters}")
        for cluster_id, faces in cluster_groups.items():
            if cluster_id == -1:
                print(f"Noise cluster: {len(faces)} faces")
            else:
                print(f"Cluster {cluster_id}: {len(faces)} faces")
        # Save all cluster assignments in a single JSON file
        cluster_json = {}
        for cluster_id, faces in cluster_groups.items():
            cluster_json[str(cluster_id)] = [face['path'] for face in faces]
        with open('face_clusters.json', 'w') as jf:
            json.dump(cluster_json, jf, indent=2)
        print("Cluster assignments saved to face_clusters.json")

        # Build Faiss index for fast similarity search
        self.build_faiss_index()

        return cluster_groups

    def build_faiss_index(self):
        """Create or update the Faiss index for similarity search"""
        if self.face_embeddings is None or len(self.face_embeddings) == 0:
            self.index = None
            return

        embeddings = self.face_embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
    
    def find_similar_faces(self, query_image_path, threshold=None, top_k=10):
        """
        Find faces similar to the query image
        Args:
            query_image_path: Path to query face image
            threshold: Similarity threshold
        Returns:
            List of similar faces with similarity scores
        """
        if not hasattr(self, 'face_embeddings') or len(self.face_embeddings) == 0:
            print("No face embeddings available. Run clustering first.")
            return []

        if threshold is None:
            threshold = self.similarity_threshold

        # Extract embedding from query image (no landmarks for external query unless provided)
        query_embedding = self.extract_face_embedding(query_image_path)
        if query_embedding is None:
            return []

        query_vec = query_embedding.astype('float32')
        faiss.normalize_L2(query_vec.reshape(1, -1))

        if self.index is not None:
            scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
            similarities = scores[0]
            idxs = indices[0]
        else:
            similarities = cosine_similarity([query_vec], self.face_embeddings)[0]
            idxs = np.argsort(similarities)[::-1][:top_k]

        # Find similar faces
        similar_faces = []
        for sim, idx in zip(similarities, idxs):
            if sim < threshold:
                continue
            dist = float(np.linalg.norm(query_vec - self.face_embeddings[idx]))
            if dist > self.distance_threshold:
                continue
            original_data_for_similar_face = next((item for item in self.extracted_faces if item['path'] == self.face_paths[idx]), None)
            similar_faces.append({
                'path': self.face_paths[idx],
                'similarity': float(sim),
                'distance': dist,
                'original_data': original_data_for_similar_face
            })
        
        # Sort by similarity
        similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_faces
    
    def save_clustering_results(self, cluster_groups, output_file="group_face_clusters.pkl"):
        """Save clustering results"""
        results = {
            'cluster_groups': cluster_groups,
            'extracted_faces': self.extracted_faces, # This list might contain entries that didn't get embeddings
            'face_embeddings': self.face_embeddings, # These are the embeddings that actually were used for clustering
            'face_paths': self.face_paths # These are the paths corresponding to face_embeddings
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Clustering results saved to {output_file}")

    def crop_and_embed_query_image(self, query_image_path, min_face_size=40, save_dir="query_crops"):
        """
        Detect faces in a query image, crop them, save the crops, and extract embeddings for each.
        Returns a list of (cropped_face_path, embedding) pairs.
        """
        import uuid
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        extracted_faces = self.detect_and_crop_faces(query_image_path, min_face_size)
        results = []
        for face_data in extracted_faces:
            face_img = face_data['image']
            if face_data.get('landmarks') is not None:
                aligned = self.align_face(face_img, face_data['landmarks'])
                if aligned is not None:
                    face_img = aligned
            crop_filename = f"query_face_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(save_dir, crop_filename)
            face_img.save(crop_path, 'JPEG', quality=95, optimize=True, progressive=True)
            embedding = self.extract_face_embedding(crop_path)
            if embedding is not None:
                results.append((crop_path, embedding))
        if not results:
            print(f"No valid face crops/embeddings extracted from {query_image_path}")
        else:
            print(f"Extracted {len(results)} face crops and embeddings from {query_image_path}")
        return results

def main():
    # Initialize the group photo processor
    processor = GroupPhotoProcessor()
    # Use the updated dataset directory
    input_directory = "/Users/pranjalsharma/Desktop/face recognition copy/group_photos"
    
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' not found.")
        print("Please create a directory with group photos or modify the path.")
        return
    
    # Min face size for overall processing. Setting to 40
    total_faces = processor.process_group_photos(input_directory, min_face_size=40) 
    
    if total_faces == 0:
        print("No faces extracted. Exiting.")
        return
    
    # Use HDBSCAN for clustering
    cluster_groups = processor.cluster_extracted_faces(min_cluster_size=2, min_samples=1)
    
    if not cluster_groups:
        print("No clusters found. Exiting.")
        return
    
    processor.save_clustering_results(cluster_groups)
    
    # Print a sample of the extracted face metadata for inspection
    print("\nSample extracted face metadata (first successfully embedded face):")
    if processor.face_paths and processor.face_embeddings.shape[0] > 0:
        # Find the original_data corresponding to the first *successfully embedded* face
        sample_path = processor.face_paths[0]
        sample_data = next((item for item in processor.extracted_faces if item['path'] == sample_path), None)
        
        if sample_data:
            print("Path:", sample_data['path'])
            print("Bounding box:", sample_data['bounding_box'])
            print("Landmarks:\n", sample_data['landmarks'])
            print("Size:", sample_data['size'])
            print("Original Image:", sample_data['original_image'])
            print("Embedding (first 10 values):", processor.face_embeddings[0][:10])
    else:
        print("No successfully extracted and embedded faces to display sample metadata.")

if __name__ == "__main__":
    main()
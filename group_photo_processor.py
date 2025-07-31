import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN
import faiss
from tqdm import tqdm
import pickle
from collections import defaultdict
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align


class GroupPhotoProcessor:
    def __init__(self, output_dir="group_faces", similarity_threshold=0.8, distance_threshold=0.7):
        """
        Initialize group photo processor

        Args:
            output_dir: Directory to save extracted faces from group photos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.distance_threshold = distance_threshold

        # Initialize detectors/encoders
        # Use insightface for primary detection/embedding and keep MTCNN for additional verification
        self.insight_app = FaceAnalysis(name="buffalo_l")
        self.insight_app.prepare(ctx_id=-1, det_size=(640, 640))
        self.mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False, min_face_size=40)
        self.rec_model = get_model("buffalo_l")
        self.rec_model.prepare(ctx_id=-1)
        # Secondary embedding model for improved accuracy
        self.facenet_model = InceptionResnetV1(pretrained="vggface2").eval()
        self.transform = T.Compose([T.Resize((160, 160)), T.ToTensor()])

        # Faiss index for fast similarity search
        self.index = None

        # Store extracted faces and their embeddings
        self.extracted_faces = []
        self.face_embeddings = []
        self.face_paths = []
        self.global_face_counter = 1  # Counter for naming faces as photo1, photo2, ...

    def detect_and_crop_faces(self, image_path, min_face_size=40):
        """Detect faces using a combination of InsightFace and MTCNN to reduce false positives."""
        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_np = np.array(img_pil)

            # Primary detection with InsightFace
            insight_faces = self.insight_app.get(img_np)
            if not insight_faces:
                print(f"No faces detected in {image_path}")
                return []

            # Secondary detection with MTCNN for verification
            mtcnn_boxes, mtcnn_probs, _ = self.mtcnn.detect(img_pil, landmarks=False)
            if mtcnn_boxes is None:
                mtcnn_boxes = np.empty((0, 4))

            extracted_faces = []
            blur_threshold = 100.0

            for i, face in enumerate(insight_faces):
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                face_width = x2 - x1
                face_height = y2 - y1
                if face_width < min_face_size or face_height < min_face_size:
                    continue

                # Verify detection with MTCNN (IoU check)
                ious = []
                for mbox in mtcnn_boxes:
                    xx1 = max(x1, mbox[0])
                    yy1 = max(y1, mbox[1])
                    xx2 = min(x2, mbox[2])
                    yy2 = min(y2, mbox[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    union = (x2 - x1) * (y2 - y1) + (mbox[2] - mbox[0]) * (mbox[3] - mbox[1]) - inter
                    if union > 0:
                        ious.append(inter / union)
                if mtcnn_boxes.size > 0 and (not ious or max(ious) < 0.3):
                    # MTCNN did not agree with this detection
                    continue

                face_img = img_pil.crop((x1, y1, x2, y2))
                face_img_cv = np.array(face_img)
                face_img_gray = cv2.cvtColor(face_img_cv, cv2.COLOR_RGB2GRAY)
                lap_var = cv2.Laplacian(face_img_gray, cv2.CV_64F).var()
                if lap_var < blur_threshold:
                    continue

                extracted_faces.append(
                    {
                        "image": face_img,
                        "box": box,
                        "landmarks": face.kps,
                        "index": i,
                        "size": (face_width, face_height),
                        "embedding": face.embedding,
                    }
                )
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
            face_img = face_data["image"]
            # Align face if landmarks are available for better embedding quality
            if face_data.get("landmarks") is not None:
                aligned = self.align_face(face_img, face_data["landmarks"])
                if aligned is not None:
                    face_img = aligned
            # Save the crop (aligned if possible)
            filename = f"photo{self.global_face_counter}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            face_img.save(filepath, "JPEG", quality=95, optimize=True, progressive=True)
            saved_paths.append(filepath)
            face_index_to_path[i] = filepath
            self.extracted_faces.append(
                {
                    "path": filepath,
                    "box": face_data["box"],
                    "bounding_box": [int(x) for x in face_data["box"]],
                    "landmarks": face_data.get("landmarks", None),
                    "size": face_data["size"],
                    "original_image": base_filename,
                    "embedding": face_data.get("embedding"),
                }
            )
            self.global_face_counter += 1
        return saved_paths

    def extract_face_embedding(self, image_path, landmarks=None):
        """Extract face embedding using a combination of ArcFace and InceptionResnetV1."""
        try:
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return None

            img = Image.open(image_path).convert("RGB")
            if landmarks is not None:
                img = Image.fromarray(face_align.norm_crop(np.array(img), np.array(landmarks)))

            # ArcFace embedding
            arc_img = img.resize((112, 112), Image.Resampling.LANCZOS)
            arc_emb = self.rec_model.get_feat(np.array(arc_img))
            if arc_emb is None:
                return None
            norm = np.linalg.norm(arc_emb)
            if norm > 0:
                arc_emb = arc_emb / norm

            # Facenet embedding
            fn_img = self.transform(img)
            with torch.no_grad():
                fn_emb = self.facenet_model(fn_img.unsqueeze(0)).squeeze(0).numpy()
            fn_norm = np.linalg.norm(fn_emb)
            if fn_norm > 0:
                fn_emb = fn_emb / fn_norm

            combined = np.concatenate([arc_emb, fn_emb])
            c_norm = np.linalg.norm(combined)
            if c_norm > 0:
                combined = combined / c_norm
            return combined
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None

    def align_face(self, img, landmarks):
        """Align face image using InsightFace utility."""
        try:
            landmarks = np.array(landmarks, dtype=np.float32)
            aligned = face_align.norm_crop(np.array(img), landmarks)
            return Image.fromarray(aligned)
        except Exception as e:
            print(f"Error aligning face: {str(e)}")
            return None

    def process_group_photos(self, input_dir, min_face_size=40):  # Adjusted default min_face_size
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
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
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
            embedding = face_data.get("embedding")
            if embedding is None:
                embedding = self.extract_face_embedding(face_data["path"], landmarks=face_data.get("landmarks"))
            if embedding is not None:
                embeddings.append(embedding)
                face_paths.append(face_data["path"])
                temp_extracted_faces_for_clustering.append(face_data)
                # Save per-face JSON file
                landmarks = face_data.get("landmarks", None)
                if isinstance(landmarks, np.ndarray):
                    landmarks = landmarks.tolist()
                json_data = {
                    "image_path": face_data["path"],  # Cropped face image path
                    "original_image": face_data.get("original_image", None),  # Original group photo path
                    "embedding": embedding.tolist(),
                    "bounding_box": [int(x) for x in face_data["box"]],
                    "landmarks": landmarks,
                }
                json_path = os.path.splitext(face_data["path"])[0] + ".json"
                with open(json_path, "w") as jf:
                    json.dump(json_data, jf, indent=2)
        if not embeddings:
            print("No valid embeddings extracted")
            return None
        self.face_embeddings = np.array(embeddings)
        self.face_paths = face_paths
        print(f"Extracted {len(embeddings)} embeddings")
        print("Clustering faces with HDBSCAN...")
        distance_matrix = cosine_distances(self.face_embeddings).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed")
        cluster_labels = clusterer.fit_predict(distance_matrix)
        from collections import defaultdict

        cluster_groups = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            original_face_data = temp_extracted_faces_for_clustering[i]
            cluster_groups[label].append(
                {
                    "path": self.face_paths[i],
                    "embedding": self.face_embeddings[i],
                    "original_data": original_face_data,
                    "bounding_box": original_face_data.get("bounding_box", None),
                    "landmarks": original_face_data.get("landmarks", None),
                }
            )
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
            cluster_json[str(cluster_id)] = [face["path"] for face in faces]
        with open("face_clusters.json", "w") as jf:
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

        embeddings = self.face_embeddings.astype("float32")
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
        if not hasattr(self, "face_embeddings") or len(self.face_embeddings) == 0:
            print("No face embeddings available. Run clustering first.")
            return []

        if threshold is None:
            threshold = self.similarity_threshold

        # Extract embedding from query image (no landmarks for external query unless provided)
        query_embedding = self.extract_face_embedding(query_image_path)
        if query_embedding is None:
            return []

        query_vec = query_embedding.astype("float32")
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
            original_data_for_similar_face = next(
                (item for item in self.extracted_faces if item["path"] == self.face_paths[idx]), None
            )
            similar_faces.append(
                {
                    "path": self.face_paths[idx],
                    "similarity": float(sim),
                    "distance": dist,
                    "original_data": original_data_for_similar_face,
                }
            )

        # Sort by similarity
        similar_faces.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_faces

    def save_clustering_results(self, cluster_groups, output_file="group_face_clusters.pkl"):
        """Save clustering results"""
        results = {
            "cluster_groups": cluster_groups,
            "extracted_faces": self.extracted_faces,  # This list might contain entries that didn't get embeddings
            "face_embeddings": self.face_embeddings,  # These are the embeddings that actually were used for clustering
            "face_paths": self.face_paths,  # These are the paths corresponding to face_embeddings
        }

        with open(output_file, "wb") as f:
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
            face_img = face_data["image"]
            if face_data.get("landmarks") is not None:
                aligned = self.align_face(face_img, face_data["landmarks"])
                if aligned is not None:
                    face_img = aligned
            crop_filename = f"query_face_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(save_dir, crop_filename)
            face_img.save(crop_path, "JPEG", quality=95, optimize=True, progressive=True)
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
        sample_data = next((item for item in processor.extracted_faces if item["path"] == sample_path), None)

        if sample_data:
            print("Path:", sample_data["path"])
            print("Bounding box:", sample_data["bounding_box"])
            print("Landmarks:\n", sample_data["landmarks"])
            print("Size:", sample_data["size"])
            print("Original Image:", sample_data["original_image"])
            print("Embedding (first 10 values):", processor.face_embeddings[0][:10])
    else:
        print("No successfully extracted and embedded faces to display sample metadata.")


if __name__ == "__main__":
    main()

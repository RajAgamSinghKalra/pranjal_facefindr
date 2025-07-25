#!/usr/bin/env python3
"""
Test script to verify cropped face images
Run this from the backend directory: python test_cropped_images.py
"""

import os
from pathlib import Path
import glob
from PIL import Image

# Your exact directory
FACE_IMAGES_DIR = Path("/Users/pranjalsharma/Desktop/face recognition copy/group_faces")

def test_cropped_images():
    print("🔍 Testing Cropped Face Images")
    print(f"Directory: {FACE_IMAGES_DIR}")
    print("=" * 60)
    
    # Check if directory exists
    if not FACE_IMAGES_DIR.exists():
        print(f"❌ Directory does not exist: {FACE_IMAGES_DIR}")
        return
    
    print(f"✅ Directory exists: {FACE_IMAGES_DIR}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
    face_images = []
    
    print("\n🔍 Scanning for image files...")
    for ext in image_extensions:
        pattern = str(FACE_IMAGES_DIR / ext)
        found_files = glob.glob(pattern)
        if found_files:
            print(f"  {ext}: {len(found_files)} files")
        face_images.extend(found_files)
    
    print(f"\n📊 Total images found: {len(face_images)}")
    
    if not face_images:
        print("\n❌ No images found!")
        print("Checking what's in the directory...")
        try:
            all_files = list(FACE_IMAGES_DIR.iterdir())
            print(f"Directory contains {len(all_files)} files:")
            for f in all_files[:20]:  # Show first 20 files
                print(f"  - {f.name} ({'directory' if f.is_dir() else f'file, {f.stat().st_size} bytes'})")
        except Exception as e:
            print(f"Error reading directory: {e}")
        return
    
    print(f"\n📸 Sample cropped face images:")
    for i, img_path in enumerate(face_images[:10]):  # Show first 10
        try:
            file_size = os.path.getsize(img_path)
            
            # Try to get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    format_info = img.format
                    dimensions = f"{width}x{height} {format_info}"
            except Exception:
                dimensions = "unknown dimensions"
            
            print(f"  {i+1:2d}. {os.path.basename(img_path)}")
            print(f"      Size: {file_size:,} bytes, {dimensions}")
            print(f"      URL:  http://localhost:8000/api/face-image/{os.path.basename(img_path)}")
            
        except Exception as e:
            print(f"  {i+1:2d}. {os.path.basename(img_path)} - Error: {e}")
    
    if len(face_images) > 10:
        print(f"  ... and {len(face_images) - 10} more images")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Start the backend server: uvicorn main:app --reload")
    print(f"2. Test the debug endpoint: http://localhost:8000/api/debug/images")
    print(f"3. Test a sample image: http://localhost:8000/api/face-image/{os.path.basename(face_images[0])}")
    print(f"4. Check the frontend at: http://localhost:3000")

if __name__ == "__main__":
    test_cropped_images()

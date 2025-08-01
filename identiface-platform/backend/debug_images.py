#!/usr/bin/env python3
"""
Debug script to check if images are accessible
Run this from the backend directory: python debug_images.py
"""

import os
from pathlib import Path
import glob

# Same paths as in main.py
ROOT_DIR = Path(__file__).resolve().parents[2]
FACE_IMAGES_DIR = Path(os.getenv("FACE_IMAGES_DIR", ROOT_DIR / "group_faces"))
GROUP_PHOTOS_DIR = Path(os.getenv("GROUP_PHOTOS_DIR", ROOT_DIR / "group_photos"))

def check_directories():
    print("🔍 Checking Image Directories...")
    print(f"Face images directory: {FACE_IMAGES_DIR}")
    print(f"Group photos directory: {GROUP_PHOTOS_DIR}")
    print()
    
    # Check if directories exist
    print("📁 Directory Status:")
    print(f"Face images dir exists: {FACE_IMAGES_DIR.exists()}")
    print(f"Group photos dir exists: {GROUP_PHOTOS_DIR.exists()}")
    print()
    
    # Check face images
    if FACE_IMAGES_DIR.exists():
        face_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']
        face_images = []
        for ext in face_extensions:
            face_images.extend(glob.glob(str(FACE_IMAGES_DIR / ext)))
        
        print(f"👥 Found {len(face_images)} face images:")
        for i, img in enumerate(face_images[:10]):  # Show first 10
            print(f"  {i+1}. {os.path.basename(img)}")
        if len(face_images) > 10:
            print(f"  ... and {len(face_images) - 10} more")
    else:
        print("❌ Face images directory not found!")
        print("💡 Create it or update the path in main.py")
    
    print()
    
    # Check group photos
    if GROUP_PHOTOS_DIR.exists():
        photo_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']
        group_photos = []
        for ext in photo_extensions:
            group_photos.extend(glob.glob(str(GROUP_PHOTOS_DIR / ext)))
        
        print(f"📸 Found {len(group_photos)} group photos:")
        for i, img in enumerate(group_photos[:10]):  # Show first 10
            print(f"  {i+1}. {os.path.basename(img)}")
        if len(group_photos) > 10:
            print(f"  ... and {len(group_photos) - 10} more")
    else:
        print("❌ Group photos directory not found!")
        print("💡 Create it or update the path in main.py")
    
    print()
    print("🔧 Next Steps:")
    if not FACE_IMAGES_DIR.exists() or not GROUP_PHOTOS_DIR.exists():
        print("1. Update the paths in backend/main.py to match your actual directories")
        print("2. Or create the directories and add some images")
    else:
        print("1. Directories look good!")
        print("2. Start the backend server: uvicorn main:app --reload")
        print("3. Test the API: http://localhost:8000/api/debug/images")

if __name__ == "__main__":
    check_directories()

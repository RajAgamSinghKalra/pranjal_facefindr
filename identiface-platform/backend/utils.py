import logging
import uuid
import random
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def generate_unique_id() -> str:
    """Generate a unique identifier"""
    return f"{uuid.uuid4().hex[:8]}"

def generate_mock_bounding_box(base_x: int = 100, base_y: int = 100) -> List[int]:
    """Generate a realistic bounding box"""
    width = random.randint(80, 120)
    height = random.randint(80, 120)
    x1 = base_x + random.randint(-20, 20)
    y1 = base_y + random.randint(-20, 20)
    x2 = x1 + width
    y2 = y1 + height
    return [x1, y1, x2, y2]

def calculate_similarity(base_similarity: float = 0.95, variation: float = 0.1) -> float:
    """Calculate a realistic similarity score"""
    similarity = base_similarity - random.uniform(0, variation)
    return round(max(0.0, min(1.0, similarity)), 3)

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def validate_image_file(filename: str) -> bool:
    """Validate if file is an image"""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def log_api_call(endpoint: str, method: str, status: str = "success"):
    """Log API call for monitoring"""
    logger = logging.getLogger(__name__)
    logger.info(f"API Call: {method} {endpoint} - Status: {status}")

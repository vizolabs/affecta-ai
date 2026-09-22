"""Facial region mapping for LRP aggregation."""

import numpy as np
from typing import Dict, Tuple


# 68-point facial landmark regions
FACIAL_REGION_LANDMARKS = {
    'mouth': list(range(48, 68)),
    'left_eye': list(range(36, 42)),
    'right_eye': list(range(42, 48)),
    'left_eyebrow': list(range(17, 22)),
    'right_eyebrow': list(range(22, 27)),
    'nose': list(range(27, 36)),
    'left_cheek': list(range(1, 9)),
    'right_cheek': list(range(9, 17)),
    'jaw': list(range(0, 17)),
}

# Mapped regions for aggregation
REGION_MAPPING = {
    'mouth': ['mouth'],
    'cheeks': ['left_cheek', 'right_cheek'],
    'eyes': ['left_eye', 'right_eye'],
    'eyebrows': ['left_eyebrow', 'right_eyebrow'],
    'nose': ['nose'],
}


def get_landmarks_for_region(
    region_name: str,
    landmarks: np.ndarray
) -> np.ndarray:
    """Get landmarks for a specific facial region.
    
    Args:
        region_name: One of 'mouth', 'cheeks', 'eyes', 'eyebrows', 'nose'
        landmarks: [68, 2] facial landmarks
    
    Returns:
        Landmarks for the specified region
    """
    if region_name not in REGION_MAPPING:
        raise ValueError(f"Unknown region: {region_name}")
    
    sub_regions = REGION_MAPPING[region_name]
    indices = []
    for sub_region in sub_regions:
        indices.extend(FACIAL_REGION_LANDMARKS[sub_region])
    
    return landmarks[indices]


def create_region_mask(
    region_name: str,
    landmarks: np.ndarray,
    height: int,
    width: int,
    radius: int = 15
) -> np.ndarray:
    """Create binary mask for a facial region.
    
    Args:
        region_name: Facial region name
        landmarks: [68, 2] facial landmarks
        height: Image height
        width: Image width
        radius: Radius around each landmark point
    
    Returns:
        Binary mask [height, width]
    """
    mask = np.zeros((height, width), dtype=np.float32)
    
    region_landmarks = get_landmarks_for_region(region_name, landmarks)
    
    for point in region_landmarks:
        x, y = int(point[0]), int(point[1])
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # Create circular region
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 <= radius**2:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        mask[ny, nx] = 1.0
    
    return mask


def aggregate_relevance_by_region(
    relevance_map: np.ndarray,
    landmarks: np.ndarray,
    regions: list = None
) -> Dict[str, float]:
    """Aggregate pixel-level relevance into facial regions.
    
    Args:
        relevance_map: [H, W] relevance heatmap
        landmarks: [68, 2] facial landmarks
        regions: List of region names (default: all)
    
    Returns:
        Dict of region -> relevance score (normalized)
    """
    if regions is None:
        regions = list(REGION_MAPPING.keys())
    
    h, w = relevance_map.shape
    region_scores = {}
    
    for region_name in regions:
        mask = create_region_mask(region_name, landmarks, h, w)
        region_scores[region_name] = np.sum(np.abs(relevance_map) * mask)
    
    # Normalize to percentages
    total = sum(region_scores.values())
    if total > 0:
        region_scores = {k: v / total for k, v in region_scores.items()}
    
    return region_scores


def get_region_center(
    region_name: str,
    landmarks: np.ndarray
) -> Tuple[int, int]:
    """Get center point of a facial region.
    
    Args:
        region_name: Facial region name
        landmarks: [68, 2] facial landmarks
    
    Returns:
        (x, y) center coordinates
    """
    region_landmarks = get_landmarks_for_region(region_name, landmarks)
    center = region_landmarks.mean(axis=0)
    return int(center[0]), int(center[1])

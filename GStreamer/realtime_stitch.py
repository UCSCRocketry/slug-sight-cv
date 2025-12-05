"""
Simple real-time stitching for stereo camera frames (no undistortion)
"""
import numpy as np


def stitch_frames(left_frame, right_frame, overlap_ratio=0.15):
    """Stitch two camera frames together with simple overlap cut
    
    Args:
        left_frame: numpy array (BGR or RGB format from camera)
        right_frame: numpy array (BGR or RGB format from camera)
        overlap_ratio: how much the images should overlap (default 15%)
    
    Returns:
        stitched_frame: numpy array of stitched image
    """
    # SWAP: "left" files are actually the right camera view, and vice versa
    # This matches the swap you had in your original code
    actual_left = right_frame
    actual_right = left_frame
    
    h, w = actual_left.shape[:2]
    
    # Calculate overlap
    overlap_pixels = int(w * overlap_ratio)
    
    # Create canvas that accounts for overlap
    canvas_width = w * 2 - overlap_pixels
    result = np.zeros((h, canvas_width, 3), dtype=np.uint8)
    
    # Place left image fully on the left
    result[:, :w] = actual_left
    
    # Calculate where right image starts (with overlap)
    right_start = w - overlap_pixels
    
    # Simple cut - place right image starting at right_start
    # This will overwrite the overlapping portion of the left image
    result[:, right_start:] = actual_right
    
    return result
"""
Optional depth estimation for enhanced 3D positioning.
Uses simple gradient-based approach for CPU efficiency on M2.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class DepthEstimator:
    """
    Lightweight depth estimation for MacBook M2.
    Uses gradient-based approach for CPU efficiency.
    """
    
    def __init__(self):
        """Initialize depth estimator."""
        self.previous_depth = None
        self.smoothing = 0.6
        print("✅ Depth Estimator initialized (CPU-optimized)")
    
    def estimate_depth(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Estimate depth map using gradient-based method.
        
        Args:
            frame: Input BGR frame
            roi: Optional region of interest (x, y, w, h)
            
        Returns:
            Dictionary with depth_map and confidence
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Gradient-based depth estimation
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Invert (higher gradient = closer = lower depth value)
        depth_map = 1.0 - depth_map
        
        # Add vertical gradient (objects higher in frame = further away)
        y_gradient = np.linspace(0.3, 0.7, h)
        y_gradient = np.tile(y_gradient[:, np.newaxis], (1, w))
        
        # Combine
        depth_map = 0.6 * depth_map + 0.4 * y_gradient
        
        # Apply smoothing
        depth_map = cv2.GaussianBlur(depth_map, (21, 21), 0)
        
        # Temporal smoothing
        if self.previous_depth is not None and depth_map.shape == self.previous_depth.shape:
            depth_map = self.smoothing * self.previous_depth + (1 - self.smoothing) * depth_map
        self.previous_depth = depth_map
        
        return {
            'depth_map': depth_map,
            'confidence': 0.5  # Moderate confidence for gradient-based method
        }
    
    def release(self):
        """Release resources."""
        self.previous_depth = None

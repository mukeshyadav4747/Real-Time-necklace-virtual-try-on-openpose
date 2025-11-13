"""
Occlusion detection for realistic rendering.
Detects hair, hands, and clothing that may occlude necklace.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class OcclusionHandler:
    """
    Simple occlusion detection using color segmentation.
    CPU-optimized for MacBook M2.
    """
    
    def __init__(self):
        """Initialize occlusion handler."""
        self.previous_mask = None
        self.smoothing = 0.7
        print("✅ Occlusion Handler initialized")
    
    def detect_occlusions(
        self,
        frame: np.ndarray,
        neck_region: Optional[Dict] = None
    ) -> Dict:
        """
        Detect occlusions using color-based segmentation.
        
        Args:
            frame: Input BGR frame
            neck_region: Optional neck tracking data
            
        Returns:
            Dictionary with occlusion_mask and has_occlusion flag
        """
        h, w = frame.shape[:2]
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Invert to get non-skin (potential occlusions)
        occlusion_mask = cv2.bitwise_not(skin_mask)
        
        # Focus on upper region if neck data available
        if neck_region and neck_region.get('collarbone_y'):
            collarbone_y = neck_region['collarbone_y']
            occlusion_mask[collarbone_y:, :] = 0
        else:
            occlusion_mask[h//2:, :] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        occlusion_mask = cv2.GaussianBlur(occlusion_mask, (7, 7), 0)
        
        # Temporal smoothing
        if self.previous_mask is not None and occlusion_mask.shape == self.previous_mask.shape:
            occlusion_mask = cv2.addWeighted(
                occlusion_mask, 1 - self.smoothing,
                self.previous_mask, self.smoothing, 0
            ).astype(np.uint8)
        self.previous_mask = occlusion_mask
        
        return {
            'occlusion_mask': occlusion_mask,
            'has_occlusion': np.any(occlusion_mask > 128)
        }
    
    def release(self):
        """Release resources."""
        self.previous_mask = None

"""
Necklace renderer with person-specific sizing and depth scaling.
Production-ready rendering with smooth transitions.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class NecklaceRenderer:
    """Professional necklace rendering with adaptive sizing."""
    
    def __init__(self, necklace_image_path: str):
        """
        Initialize necklace renderer.
        
        Args:
            necklace_image_path: Path to transparent PNG necklace image
        """
        self.necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
        
        if self.necklace_image is None:
            raise ValueError(f"Could not load necklace: {necklace_image_path}")
        
        # Ensure RGBA format
        if self.necklace_image.shape[2] == 3:
            alpha_channel = np.ones(
                (self.necklace_image.shape[0], self.necklace_image.shape[1], 1),
                dtype=self.necklace_image.dtype
            ) * 255
            self.necklace_image = np.concatenate([self.necklace_image, alpha_channel], axis=2)
        
        self.original_height, self.original_width = self.necklace_image.shape[:2]
        self.aspect_ratio = self.original_width / self.original_height
        
        # Smoothing buffers
        self.previous_size = None
        self.previous_position = None
        self.size_smoothing = 0.70
        self.position_smoothing = 0.60
        
        print(f"✅ Necklace loaded: {self.necklace_image.shape}")
    
    def load_necklace(self, necklace_image_path: str) -> bool:
        """
        Load a new necklace image dynamically.
        
        Args:
            necklace_image_path: Path to new necklace image
            
        Returns:
            True if successful, False otherwise
        """
        new_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)
        
        if new_image is None:
            print(f"⚠️ Could not load: {necklace_image_path}")
            return False
        
        if new_image.shape[2] == 3:
            alpha = np.ones((new_image.shape[0], new_image.shape[1], 1),
                          dtype=new_image.dtype) * 255
            new_image = np.concatenate([new_image, alpha], axis=2)
        
        self.necklace_image = new_image
        self.original_height, self.original_width = self.necklace_image.shape[:2]
        self.aspect_ratio = self.original_width / self.original_height
        
        # Reset smoothing
        self.previous_size = None
        self.previous_position = None
        
        print(f"✅ Necklace changed to: {necklace_image_path}")
        return True
    
    def render_necklace(
        self,
        frame: np.ndarray,
        neck_data: Dict
    ) -> np.ndarray:
        """
        Render necklace on frame with person-specific sizing.
        
        Args:
            frame: Input BGR frame
            neck_data: Dictionary from pose_tracker with neck information
            
        Returns:
            Frame with rendered necklace
        """
        if not neck_data['neck_detected'] or not neck_data['necklace_curve_2d']:
            self.previous_size = None
            self.previous_position = None
            return frame
        
        curve_2d = neck_data['necklace_curve_2d']
        curve_3d = neck_data['necklace_curve_3d']
        shoulder_width = neck_data['shoulder_width']
        collarbone_y = neck_data['collarbone_y']
        
        # Calculate base size from shoulder width
        base_width = int(shoulder_width * 0.62)
        
        # Apply depth-based scaling if 3D data available
        if curve_3d and len(curve_3d) > 0:
            center_idx = len(curve_3d) // 2
            center_region = curve_3d[max(0, center_idx-5):min(len(curve_3d), center_idx+5)]
            avg_depth_z = np.mean([p[2] for p in center_region])
            
            # Depth scaling: further = smaller, closer = larger
            depth_scale = 1.0 - (avg_depth_z * 1.5)
            depth_scale = np.clip(depth_scale, 0.80, 1.20)
        else:
            depth_scale = 1.0
        
        # Calculate final size
        target_width = int(base_width * depth_scale)
        target_height = int(target_width / self.aspect_ratio)
        
        # Apply size smoothing
        if self.previous_size is not None:
            target_width = int(
                target_width * (1 - self.size_smoothing) +
                self.previous_size[0] * self.size_smoothing
            )
            target_height = int(
                target_height * (1 - self.size_smoothing) +
                self.previous_size[1] * self.size_smoothing
            )
        self.previous_size = (target_width, target_height)
        
        # Resize necklace
        resized_necklace = cv2.resize(
            self.necklace_image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Calculate position at collarbone level
        curve_xs = [p[0] for p in curve_2d]
        center_x = int(np.mean(curve_xs))
        
        x = center_x - target_width // 2
        y = collarbone_y - target_height // 2
        
        # Apply position smoothing
        if self.previous_position is not None:
            x = int(x * (1 - self.position_smoothing) + 
                   self.previous_position[0] * self.position_smoothing)
            y = int(y * (1 - self.position_smoothing) + 
                   self.previous_position[1] * self.position_smoothing)
        self.previous_position = (x, y)
        
        # Overlay necklace
        output = self._overlay_with_alpha(frame, resized_necklace, x, y)
        
        return output
    
    def _overlay_with_alpha(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """
        Alpha blend overlay onto background.
        
        Args:
            background: Background frame
            overlay: Overlay image with alpha channel
            x, y: Top-left position for overlay
            
        Returns:
            Blended frame
        """
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        # Calculate valid region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)
        
        ov_x1, ov_y1 = max(0, -x), max(0, -y)
        ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return background
        
        # Extract regions
        bg_region = background[y1:y2, x1:x2].astype(float)
        ov_region = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        
        # Alpha blending
        if ov_region.shape[2] == 4:
            ov_rgb = ov_region[:, :, :3].astype(float)
            ov_alpha = ov_region[:, :, 3:].astype(float) / 255.0
            
            blended = (ov_alpha * ov_rgb + (1 - ov_alpha) * bg_region).astype(np.uint8)
            background[y1:y2, x1:x2] = blended
        
        return background

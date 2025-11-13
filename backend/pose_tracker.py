"""
FIXED: Accurate neck positioning for necklace placement.
The necklace will now sit at the proper collarbone level.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
import os

class PoseTracker:
    """OpenCV DNN-based pose tracking with ACCURATE collarbone positioning."""
    
    def __init__(self, model_folder: str = "backend/models"):
        """Initialize OpenCV DNN with Caffe models."""
        print("🚀 Initializing OpenCV DNN Pose Tracker...")
        
        # Model files
        self.prototxt = os.path.join(model_folder, "pose_deploy.prototxt")
        self.caffemodel = os.path.join(model_folder, "pose_iter_584000.caffemodel")
        
        if not os.path.exists(self.prototxt):
            raise FileNotFoundError(f"Prototxt not found: {self.prototxt}")
        if not os.path.exists(self.caffemodel):
            raise FileNotFoundError(f"Caffemodel not found: {self.caffemodel}")
        
        # Load network
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # BODY_25 keypoint indices
        self.NECK = 1
        self.RIGHT_SHOULDER = 2
        self.LEFT_SHOULDER = 5
        self.NOSE = 0
        self.RIGHT_EAR = 17
        self.LEFT_EAR = 18
        
        # Detection parameters
        self.input_width = 368
        self.input_height = 368
        self.threshold = 0.1
        
        # Smoothing
        self.previous_keypoints = None
        self.previous_curve = None
        self.previous_collarbone_y = None
        self.smoothing_factor = 0.70
        
        print(f"✅ Pose Tracker initialized!")
    
    def detect_neck_region(self, frame: np.ndarray) -> Dict:
        """Detect neck region with ACCURATE collarbone positioning."""
        h, w, _ = frame.shape
        
        # Prepare input
        inp_blob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, 
            (self.input_width, self.input_height),
            (0, 0, 0), swapRB=False, crop=False
        )
        
        # Forward pass
        self.net.setInput(inp_blob)
        output = self.net.forward()
        
        # Default output
        result = {
            'neck_detected': False,
            'necklace_curve_2d': [],
            'necklace_curve_3d': [],
            'collarbone_y': 0,
            'shoulder_width': 0,
            'neck_width': 0,
            'keypoints': None,
            'confidence': 0.0,
            'annotated_frame': frame.copy()
        }
        
        # Extract keypoints
        keypoints = self._extract_keypoints(output, w, h)
        
        if keypoints is None:
            return result
        
        # Apply smoothing
        if self.previous_keypoints is not None:
            keypoints = self._smooth_keypoints(keypoints, self.previous_keypoints)
        self.previous_keypoints = keypoints.copy()
        
        # Get critical points
        neck = keypoints[self.NECK]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        nose = keypoints[self.NOSE]
        
        # Check detection
        if neck[2] < self.threshold:
            return result
        
        neck_2d = (int(neck[0]), int(neck[1]))
        right_shoulder_2d = (int(right_shoulder[0]), int(right_shoulder[1]))
        left_shoulder_2d = (int(left_shoulder[0]), int(left_shoulder[1]))
        nose_2d = (int(nose[0]), int(nose[1]))
        
        # Calculate shoulder width
        shoulder_width = float(np.linalg.norm(
            np.array(left_shoulder_2d) - np.array(right_shoulder_2d)
        ))
        neck_width = shoulder_width * 0.30
        
        # CRITICAL FIX: Calculate ACTUAL collarbone position
        # In BODY_25, keypoint 1 (neck) is actually at the neck BASE (near collarbone)
        # But we need to position the necklace SLIGHTLY BELOW this point
        
        # Method 1: Use distance from nose to neck as reference
        nose_to_neck_dist = np.linalg.norm(np.array(nose_2d) - np.array(neck_2d))
        
        # Collarbone is approximately 20-25% of nose-to-neck distance BELOW the neck keypoint
        collarbone_offset = int(nose_to_neck_dist * 0.25)
        collarbone_y = int(neck_2d[1] + collarbone_offset)
        
        # Apply temporal smoothing to collarbone_y
        if self.previous_collarbone_y is not None:
            collarbone_y = int(0.3 * collarbone_y + 0.7 * self.previous_collarbone_y)
        self.previous_collarbone_y = collarbone_y
        
        # Generate necklace curve
        curve_2d, curve_3d = self._generate_necklace_curve(
            neck_2d, left_shoulder_2d, right_shoulder_2d, 
            nose_2d, collarbone_y, w, h
        )
        
        # Apply curve smoothing
        if self.previous_curve and len(curve_2d) == len(self.previous_curve):
            curve_2d = self._smooth_curve(curve_2d, self.previous_curve)
        self.previous_curve = curve_2d.copy()
        
        # Update result
        result.update({
            'neck_detected': True,
            'necklace_curve_2d': curve_2d,
            'necklace_curve_3d': curve_3d,
            'collarbone_y': collarbone_y,
            'shoulder_width': shoulder_width,
            'neck_width': neck_width,
            'keypoints': keypoints,
            'confidence': float(neck[2]),
            'annotated_frame': self._draw_debug(frame, keypoints, curve_2d, collarbone_y)
        })
        
        return result
    
    def _generate_necklace_curve(
        self,
        neck: Tuple[int, int],
        left_shoulder: Tuple[int, int],
        right_shoulder: Tuple[int, int],
        nose: Tuple[int, int],
        collarbone_y: int,
        w: int,
        h: int,
        num_points: int = 70
    ) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """
        Generate accurate necklace curve at collarbone level.
        FIXED: Proper positioning using nose-to-neck reference.
        """
        
        # Calculate center and span
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        shoulder_width = np.linalg.norm(
            np.array(left_shoulder) - np.array(right_shoulder)
        )
        
        # Necklace span: 55% of shoulder width (tighter fit)
        necklace_span = shoulder_width * 0.55
        
        # Necklace endpoints (at neck sides, not at shoulders)
        left_endpoint_x = int(shoulder_center_x - necklace_span / 2)
        right_endpoint_x = int(shoulder_center_x + necklace_span / 2)
        
        # CRITICAL: Endpoint Y-position
        # Endpoints should be at collarbone level (not higher)
        # They curve slightly upward at the ends to wrap around neck
        nose_to_neck_dist = abs(nose[1] - neck[1])
        endpoint_y_raise = int(nose_to_neck_dist * 0.15)  # Slight raise at ends
        
        left_endpoint_y = collarbone_y - endpoint_y_raise
        right_endpoint_y = collarbone_y - endpoint_y_raise
        
        # Center drape (lowest point)
        # Center hangs 10-15% lower than collarbone
        center_drape = int(nose_to_neck_dist * 0.12)
        center_y = collarbone_y + center_drape
        
        # 9-point control curve for realistic shape
        control_points = [
            # Left endpoint
            (left_endpoint_x, left_endpoint_y),
            
            # Left outer curve
            (int(left_endpoint_x + necklace_span * 0.12), 
             int(left_endpoint_y + center_drape * 0.25)),
            
            # Left mid
            (int(left_endpoint_x + necklace_span * 0.22), 
             int(collarbone_y)),
            
            # Left inner
            (int(left_endpoint_x + necklace_span * 0.38), 
             int(collarbone_y + center_drape * 0.70)),
            
            # Center (lowest point)
            (int(shoulder_center_x), int(center_y)),
            
            # Right inner
            (int(right_endpoint_x - necklace_span * 0.38), 
             int(collarbone_y + center_drape * 0.70)),
            
            # Right mid
            (int(right_endpoint_x - necklace_span * 0.22), 
             int(collarbone_y)),
            
            # Right outer curve
            (int(right_endpoint_x - necklace_span * 0.12), 
             int(right_endpoint_y + center_drape * 0.25)),
            
            # Right endpoint
            (right_endpoint_x, right_endpoint_y)
        ]
        
        # Generate 3D points
        control_points_3d = []
        for i, (x, y) in enumerate(control_points):
            t = i / (len(control_points) - 1)
            z = 0.06 - 0.03 * (4 * (t - 0.5) ** 2)
            control_points_3d.append(np.array([x / w, y / h, z]))
        
        # Cubic spline interpolation
        x_pts = [p[0] for p in control_points]
        y_pts = [p[1] for p in control_points]
        
        t = np.linspace(0, 1, len(control_points))
        t_new = np.linspace(0, 1, num_points)
        
        try:
            cs_x = CubicSpline(t, x_pts, bc_type='natural')
            cs_y = CubicSpline(t, y_pts, bc_type='natural')
            
            smooth_x = cs_x(t_new)
            smooth_y = cs_y(t_new)
            
            curve_2d = [(int(x), int(y)) for x, y in zip(smooth_x, smooth_y)]
            
            curve_3d = []
            for i, t_val in enumerate(t_new):
                idx = int(t_val * (len(control_points_3d) - 1))
                idx = min(idx, len(control_points_3d) - 1)
                z = control_points_3d[idx][2]
                curve_3d.append(np.array([smooth_x[i] / w, smooth_y[i] / h, z]))
        except:
            curve_2d = control_points
            curve_3d = control_points_3d
        
        return curve_2d, curve_3d
    
    def _extract_keypoints(self, output, frame_width, frame_height):
        """Extract keypoints from network output."""
        H = output.shape[2]
        W = output.shape[3]
        
        keypoints = []
        
        for i in range(output.shape[1]):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H
            
            if prob > self.threshold:
                keypoints.append([x, y, prob])
            else:
                keypoints.append([0, 0, 0])
        
        keypoints = np.array(keypoints)
        
        if keypoints[self.NECK][2] < self.threshold:
            return None
        
        return keypoints
    
    def _smooth_keypoints(self, current, previous):
        """Temporal smoothing."""
        alpha = 1 - self.smoothing_factor
        return alpha * current + self.smoothing_factor * previous
    
    def _smooth_curve(self, current, previous):
        """Curve smoothing."""
        smoothed = []
        alpha = 1 - self.smoothing_factor
        for curr, prev in zip(current, previous):
            sx = int(alpha * curr[0] + self.smoothing_factor * prev[0])
            sy = int(alpha * curr[1] + self.smoothing_factor * prev[1])
            smoothed.append((sx, sy))
        return smoothed
    
    def _draw_debug(self, frame, keypoints, curve, collarbone_y):
        """Draw debug visualization."""
        output = frame.copy()
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > self.threshold:
                cv2.circle(output, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                # Label important points
                if i == self.NECK:
                    cv2.putText(output, "NECK", (int(kp[0])+10, int(kp[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                elif i in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER]:
                    cv2.putText(output, "SHOULDER", (int(kp[0])+10, int(kp[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw skeleton
        pose_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7]]
        for pair in pose_pairs:
            part_a = keypoints[pair[0]]
            part_b = keypoints[pair[1]]
            if part_a[2] > self.threshold and part_b[2] > self.threshold:
                cv2.line(output,
                        (int(part_a[0]), int(part_a[1])),
                        (int(part_b[0]), int(part_b[1])),
                        (0, 255, 255), 2)
        
        # Draw necklace curve
        if len(curve) > 1:
            for i in range(len(curve) - 1):
                cv2.line(output, curve[i], curve[i+1], (255, 0, 255), 3)
        
        # Draw collarbone reference line
        cv2.line(output, (0, collarbone_y), (output.shape[1], collarbone_y), 
                (0, 255, 0), 1)
        cv2.putText(output, "COLLARBONE", (10, collarbone_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output
    
    def release(self):
        """Release resources."""
        print("✅ Pose Tracker released")

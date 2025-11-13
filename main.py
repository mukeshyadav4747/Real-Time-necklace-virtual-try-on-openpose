"""
FastAPI backend for necklace virtual try-on.
Uses OpenCV DNN with Caffe models for pose detection.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import json
import os
from glob import glob

from backend.pose_tracker import PoseTracker
from backend.necklace_renderer import NecklaceRenderer
from backend.depth_estimator import DepthEstimator
from backend.occlusion_handler import OcclusionHandler

app = FastAPI(
    title="Virtual Necklace Try-On API",
    description="Production-ready necklace virtual try-on using OpenCV DNN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
print("🚀 Initializing Virtual Necklace Try-On API...")

try:
    # Initialize pose tracker with Caffe models
    pose_tracker = PoseTracker(model_folder="backend/models")
    
    # Find necklace images
    necklace_dir = "backend/assets"
    os.makedirs(necklace_dir, exist_ok=True)
    
    necklace_files = glob(os.path.join(necklace_dir, "*.png"))
    necklace_files.extend(glob(os.path.join(necklace_dir, "*.jpg")))
    
    if necklace_files:
        necklace_renderer = NecklaceRenderer(necklace_files[0])
        print(f"✅ Loaded necklace: {necklace_files[0]}")
        print(f"✅ Found {len(necklace_files)} necklace images")
    else:
        print(f"⚠️ No necklace images in {necklace_dir}")
        necklace_renderer = None
    
    # Optional depth estimator
    depth_estimator = DepthEstimator()
    
    # Optional occlusion handler
    occlusion_handler = OcclusionHandler()
    
    print("✅ System initialized successfully!")
    
except Exception as e:
    print(f"❌ Initialization error: {e}")
    pose_tracker = None
    necklace_renderer = None
    depth_estimator = None
    occlusion_handler = None

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "status": "online" if pose_tracker and necklace_renderer else "error",
        "message": "Virtual Necklace Try-On API",
        "model": "OpenCV DNN with Caffe (BODY_25)",
        "platform": "MacBook M2 Compatible",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pose_tracker": pose_tracker is not None,
        "necklace_renderer": necklace_renderer is not None
    }

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing."""
    await websocket.accept()
    print("✅ Client connected to WebSocket")
    
    if not pose_tracker or not necklace_renderer:
        await websocket.send_text(json.dumps({
            "error": "System not initialized. Check models and necklace images."
        }))
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            try:
                frame_data = json.loads(data)
                frame_b64 = frame_data.get('frame', '')
                
                # Decode frame
                frame_bytes = base64.b64decode(
                    frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64
                )
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame
                neck_data = pose_tracker.detect_neck_region(frame)
                output_frame = necklace_renderer.render_necklace(frame, neck_data)
                
                # Encode result
                _, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64_out = base64.b64encode(buffer).decode('utf-8')
                
                # Send response
                response = {
                    'frame': f"data:image/jpeg;base64,{frame_b64_out}",
                    'neck_detected': neck_data['neck_detected'],
                    'confidence': neck_data['confidence'],
                    'model': 'OpenCV DNN (Caffe)'
                }
                
                await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON received")
                continue
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
                
    except WebSocketDisconnect:
        print("❌ Client disconnected from WebSocket")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/video")
    print("📄 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

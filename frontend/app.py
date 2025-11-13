"""
Streamlit UI for Virtual Necklace Try-On
Production-ready interface with necklace selection and real-time preview
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import sys
import os
import time
from glob import glob

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pose_tracker import PoseTracker
from backend.necklace_renderer import NecklaceRenderer

# Page configuration
st.set_page_config(
    page_title="💎 Virtual Necklace Try-On",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">💎 Virtual Necklace Try-On</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-Ready AR Jewelry Solution | OpenCV DNN | MacBook M2 Optimized</p>', unsafe_allow_html=True)
st.markdown("---")

# Global state
if 'selected_necklace' not in st.session_state:
    st.session_state.selected_necklace = None
if 'necklace_changed' not in st.session_state:
    st.session_state.necklace_changed = False

# Initialize system
@st.cache_resource
def init_system():
    """Initialize pose tracker and necklace renderer."""
    try:
        # Check if models exist
        model_folder = "backend/models"
        prototxt = os.path.join(model_folder, "pose_deploy.prototxt")
        caffemodel = os.path.join(model_folder, "pose_iter_584000.caffemodel")
        
        if not os.path.exists(prototxt):
            st.error(f"""
            ❌ **Missing Model File**
            
            Required: `{prototxt}`
            
            **Download:**
            ```
            mkdir -p backend/models
            cd backend/models
            wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
            wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
            ```
            """)
            return None, None, []
        
        if not os.path.exists(caffemodel):
            st.error(f"❌ Missing: {caffemodel}")
            return None, None, []
        
        # Initialize tracker
        pose_tracker = PoseTracker(model_folder=model_folder)
        
        # Get necklace images
        necklace_dir = "backend/assets"
        os.makedirs(necklace_dir, exist_ok=True)
        
        necklace_files = glob(os.path.join(necklace_dir, "*.png"))
        necklace_files.extend(glob(os.path.join(necklace_dir, "*.jpg")))
        
        if not necklace_files:
            st.warning(f"""
            ⚠️ **No Necklace Images Found**
            
            Add transparent PNG images to: `{necklace_dir}`
            
            **Example:**
            ```
            cp your_necklace.png backend/assets/
            ```
            """)
            return pose_tracker, None, []
        
        # Initialize renderer
        necklace_renderer = NecklaceRenderer(necklace_files[0])
        necklace_names = [os.path.basename(f) for f in necklace_files]
        
        st.success(f"✅ System initialized! Found {len(necklace_files)} necklaces")
        
        return pose_tracker, necklace_renderer, necklace_names
        
    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, []

pose_tracker, necklace_renderer, necklace_options = init_system()

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Necklace selection
    st.markdown("### 💍 Necklace Selection")
    if necklace_options:
        selected_necklace = st.selectbox(
            "Choose your design:",
            necklace_options,
            key="necklace_selector"
        )
        
        # Detect necklace change
        if st.session_state.selected_necklace != selected_necklace:
            st.session_state.selected_necklace = selected_necklace
            st.session_state.necklace_changed = True
            
            # Load new necklace
            if necklace_renderer:
                necklace_path = os.path.join("backend", "assets", selected_necklace)
                if necklace_renderer.load_necklace(necklace_path):
                    st.success(f"✅ Switched to: {selected_necklace}")
                else:
                    st.error("❌ Failed to load necklace")
    else:
        st.warning("No necklaces available. Add PNG files to backend/assets/")
    
    st.markdown("---")
    
    # Display settings
    st.markdown("### 🎨 Display Settings")
    show_keypoints = st.checkbox("Show Pose Keypoints", value=False, help="Show detected body keypoints")
    show_stats = st.checkbox("Show Statistics", value=True, help="Display FPS and detection info")
    mirror_mode = st.checkbox("Mirror Mode", value=True, help="Mirror the camera view")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📋 How to Use")
    st.info("""
    **Setup:**
    1. Select necklace from dropdown above
    2. Click START button below
    3. Allow camera access when prompted
    
    **Best Results:**
    ✓ Good lighting (front/top light)
    ✓ Plain background
    ✓ Face camera directly
    ✓ Keep shoulders visible
    ✓ Stand 1-2 meters from camera
    
    **Tips:**
    • Necklace auto-adjusts to your size
    • Move slowly for stable rendering
    • Try different angles!
    """)
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 🤖 Model Information")
    st.code("""
Model: OpenCV DNN (Caffe)
Architecture: BODY_25
Platform: MacBook M2 (CPU)
Expected FPS: 12-20
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # WebRTC configuration
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Video processor
    class VideoProcessor:
        def __init__(self):
            self.pose_tracker = pose_tracker
            self.necklace_renderer = necklace_renderer
            self.fps_list = []
            self.last_time = time.time()
            self.frame_count = 0
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Process each video frame."""
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror mode
            if mirror_mode:
                img = cv2.flip(img, 1)
            
            # Check if system initialized
            if not self.pose_tracker or not self.necklace_renderer:
                cv2.putText(img, "System not initialized", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, "Check models and necklace images", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Detect neck region
            neck_data = self.pose_tracker.detect_neck_region(img)
            
            # Render necklace
            if show_keypoints:
                output = self.necklace_renderer.render_necklace(
                    neck_data['annotated_frame'], neck_data
                )
            else:
                output = self.necklace_renderer.render_necklace(img, neck_data)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time + 0.001)
            self.last_time = current_time
            
            self.fps_list.append(fps)
            if len(self.fps_list) > 20:
                self.fps_list.pop(0)
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            
            # Display stats
            if show_stats:
                # Semi-transparent background
                overlay = output.copy()
                cv2.rectangle(overlay, (5, 5), (350, 130), (0, 0, 0), -1)
                output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
                
                stats = [
                    f"FPS: {avg_fps:.1f}",
                    f"Neck: {'DETECTED' if neck_data['neck_detected'] else 'NOT FOUND'}",
                    f"Confidence: {neck_data['confidence']:.2f}",
                    f"Model: OpenCV DNN (Caffe)"
                ]
                
                for i, text in enumerate(stats):
                    color = (0, 255, 0) if avg_fps > 10 else (0, 255, 255)
                    cv2.putText(output, text, (15, 35 + i*28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(output, format="bgr24")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="necklace-tryon-opencv-dnn-v1",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True
    )

with col2:
    st.subheader("📊 Status Panel")
    
    # Connection status
    status_placeholder = st.empty()
    if webrtc_ctx.state.playing:
        status_placeholder.markdown('<div class="stats-box">🟢 Camera Active</div>', unsafe_allow_html=True)
    else:
        status_placeholder.warning("🔴 Click START to begin")
    
    st.markdown("---")
    
    # Controls
    st.subheader("🎮 Controls")
    
    if st.button("🔄 Restart System", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("📸 Take Screenshot", use_container_width=True):
        st.info("Screenshot feature coming soon!")
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("📈 Performance")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Target FPS", "12-20", help="Expected frames per second")
    with metric_col2:
        st.metric("Latency", "<50ms", help="Processing delay")
    
    st.markdown("---")
    
    # Current necklace display
    st.subheader("💎 Active Necklace")
    if st.session_state.selected_necklace:
        st.success(st.session_state.selected_necklace)
        
        # Show necklace preview if image exists
        necklace_path = os.path.join("backend", "assets", st.session_state.selected_necklace)
        if os.path.exists(necklace_path):
            try:
                import PIL.Image as Image
                img = Image.open(necklace_path)
                st.image(img, caption="Preview", use_column_width=True)
            except:
                pass
    else:
        st.info("No necklace selected")
    
    st.markdown("---")
    
    # Quick tips
    st.subheader("💡 Quick Tips")
    st.markdown("""
    - **Lighting**: Bright, even lighting works best
    - **Distance**: Stay 1-2 meters from camera
    - **Movement**: Slow movements = stable tracking
    - **Background**: Plain backgrounds improve detection
    """)

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("💎 Virtual Necklace Try-On")
with col_f2:
    st.caption("🤖 OpenCV DNN (Caffe Models)")
with col_f3:
    st.caption("🖥️ MacBook M2 Compatible")

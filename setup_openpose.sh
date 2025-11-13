#!/bin/bash

# OpenPose Installation Script with CUDA
# For Ubuntu/Linux systems

echo "🚀 Installing OpenPose with CUDA support..."

# Update system
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev

# Install CUDA (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Clone OpenPose
echo "📥 Cloning OpenPose..."
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose

# Download models
echo "📥 Downloading models..."
bash scripts/ubuntu/install_deps.sh

# Build with CUDA and Python bindings
echo "🔨 Building OpenPose with CUDA..."
mkdir build && cd build

cmake -D GPU_MODE=CUDA \
      -D BUILD_PYTHON=ON \
      -D PYTHON_EXECUTABLE=$(which python3) \
      -D PYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      ..

make -j`nproc`

# Set environment variable
echo "export OPENPOSE_PATH=$(pwd)/.." >> ~/.bashrc
source ~/.bashrc

echo "✅ OpenPose installation complete!"
echo "Run: export OPENPOSE_PATH=/path/to/openpose"

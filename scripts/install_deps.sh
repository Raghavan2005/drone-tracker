#!/bin/bash
set -e

echo "Installing system dependencies for drone-tracker..."

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-dev \
    python3-pip \
    python3-venv

echo ""
echo "System dependencies installed."
echo ""
echo "For GPU support, install:"
echo "  - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
echo "  - TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
echo "  - ONNX Runtime (GPU): pip install onnxruntime-gpu"
echo ""
echo "To set up the Python training environment:"
echo "  cd training && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"

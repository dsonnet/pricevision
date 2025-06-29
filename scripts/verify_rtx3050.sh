#!/bin/bash

echo "🔍 Verifying RTX 3050 System Requirements..."

# Check NVIDIA driver
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA driver not found"
    exit 1
fi

# Verify RTX 3050 specifically
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

echo "🎮 Detected GPU: $GPU_NAME"
echo "💾 GPU Memory: ${GPU_MEMORY}MB"

if [[ "$GPU_NAME" == *"RTX 3050"* ]] && [[ "$GPU_MEMORY" -ge 4000 ]]; then
    echo "✅ RTX 3050 verified - 4GB VRAM available"
else
    echo "⚠️  Warning: Expected RTX 3050 with 4GB VRAM"
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "🔧 CUDA Version: $CUDA_VERSION"
fi

# Check system resources
echo "🖥️  System RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "💿 Available Disk: $(df -h / | awk 'NR==2 {print $4}')"

echo "✅ System verification complete"
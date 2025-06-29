#!/usr/bin/env python3
"""
RTX 3050 Setup Verification Script
Tests GPU availability, memory constraints, and key libraries for PriceVision
"""

import torch
import subprocess
import psutil
import sys
import os
from pathlib import Path

def verify_rtx3050_setup():
    """Comprehensive RTX 3050 setup verification"""
    print("🔍 RTX 3050 Setup Verification")
    print("=" * 50)
    
    # Basic CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(0)
        
        print(f"🎮 GPU Device: {device_name}")
        print(f"🔢 Device Count: {device_count}")
        print(f"📍 Current Device: {current_device}")
        
        # Check memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024**3)
        print(f"💾 Total GPU Memory: {total_memory_gb:.1f} GB")
        
        # Verify RTX 3050 specifically
        if "RTX 3050" in device_name and total_memory_gb >= 3.8:
            print("✅ RTX 3050 verified successfully")
        else:
            print("⚠️  Warning: Expected RTX 3050 with ~4GB memory")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            allocated = torch.cuda.memory_allocated(0)
            print(f"🧪 Test allocation: {allocated / (1024**2):.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ Memory allocation test passed")
        except RuntimeError as e:
            print(f"❌ Memory allocation failed: {e}")
    
    # System resources
    ram = psutil.virtual_memory()
    print(f"🖥️  System RAM: {ram.total / (1024**3):.1f} GB ({ram.percent}% used)")
    
    # Test key libraries
    print("\n🔧 Testing Key Libraries:")
    
    # Test EasyOCR (CPU-based for RTX 3050)
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'fr'], gpu=False)  # CPU for RTX 3050
        print("✅ EasyOCR initialized (CPU mode)")
    except Exception as e:
        print(f"❌ EasyOCR failed: {e}")
    
    # Test lightweight transformer
    try:
        from transformers import pipeline
        # Use small model for RTX 3050
        classifier = pipeline("text-classification", 
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            device=0 if cuda_available else -1)
        result = classifier("This is a test")
        print("✅ Transformers pipeline working")
    except Exception as e:
        print(f"❌ Transformers failed: {e}")
    
    # Test sentence transformers (lightweight)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence"])
        print("✅ Sentence Transformers working")
    except Exception as e:
        print(f"❌ Sentence Transformers failed: {e}")
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV failed: {e}")
    
    # Test database connectivity
    try:
        import mysql.connector
        print("✅ MySQL connector available")
    except Exception as e:
        print(f"❌ MySQL connector failed: {e}")
    
    print("\n🎯 RTX 3050 Optimization Settings:")
    print(f"   - Max GPU Memory Usage: 3.5GB (87.5% of 4GB)")
    print(f"   - Concurrent Requests: 5 maximum")
    print(f"   - Model Loading: Dynamic (no preloading)")
    print(f"   - OCR Processing: CPU-based")
    print(f"   - Embedding Models: CPU-based")
    
    # Environment variables check
    print("\n🌍 Environment Variables:")
    env_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_VISIBLE_DEVICES',
        'OMP_NUM_THREADS',
        'TOKENIZERS_PARALLELISM'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    return cuda_available

def test_memory_management():
    """Test RTX 3050 memory management strategies"""
    print("\n🧠 Testing Memory Management:")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping memory tests")
        return
    
    try:
        # Test gradual memory allocation
        print("📈 Testing gradual memory allocation...")
        tensors = []
        
        for i in range(10):
            tensor = torch.randn(100, 100).cuda()
            tensors.append(tensor)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"   Step {i+1}: {allocated:.1f} MB allocated")
            
            if allocated > 3500:  # 3.5GB limit for RTX 3050
                print("⚠️  Approaching memory limit, cleaning up...")
                break
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        print("✅ Memory management test completed")
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")

def main():
    """Main verification function"""
    print("🚀 Starting RTX 3050 PriceVision Setup Verification\n")
    
    # Run basic setup verification
    cuda_available = verify_rtx3050_setup()
    
    # Run memory management tests
    if cuda_available:
        test_memory_management()
    
    print("\n" + "=" * 50)

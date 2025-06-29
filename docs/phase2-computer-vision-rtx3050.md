# Phase 2: Computer Vision Integration - RTX 3050 Optimized (3-4 weeks)

## Overview
This phase implements computer vision capabilities optimized for RTX 3050 with 4GB VRAM, focusing on lightweight models and efficient processing pipelines.

## Hardware Constraints & Solutions
- **VRAM Limit**: 4GB total, ~3GB usable for models
- **Strategy**: Use quantized models, CPU fallbacks, and model swapping
- **Performance**: Prioritize accuracy over speed for this hardware tier

## Week 1: Lightweight OCR Pipeline

### OCR Engine Selection (VRAM-Optimized)
**Primary**: EasyOCR with CPU processing
**Secondary**: Surya OCR (lightweight version)
**Fallback**: Tesseract for basic text extraction

```python
# ocr_pipeline.py - RTX 3050 Optimized
import easyocr
import cv2
import torch
from PIL import Image
import numpy as np

class OptimizedOCRPipeline:
    def __init__(self):
        # Force CPU for OCR to save VRAM
        self.easyocr_reader = easyocr.Reader(['en', 'fr'], gpu=False)
        
        # Only use GPU for preprocessing if VRAM available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_memory_threshold = 1024  # MB reserved for other models
        
    def check_gpu_memory(self):
        """Check available GPU memory before processing"""
        if torch.cuda.is_available():
            memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return memory_free > (self.gpu_memory_threshold * 1024 * 1024)
        return False
    
    def preprocess_image(self, image_path):
        """Lightweight image preprocessing"""
        image = cv2.imread(image_path)
        
        # Basic preprocessing without GPU-intensive operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_text(self, image_path):
        """Extract text using CPU-based OCR"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Use EasyOCR on CPU to preserve VRAM
            results = self.easyocr_reader.readtext(processed_image)
            
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low-confidence results
                    extracted_text.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            return extracted_text
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return []
    
    def extract_game_title(self, ocr_results):
        """Extract likely game title from OCR results"""
        if not ocr_results:
            return None
            
        # Sort by confidence and position (top of image likely contains title)
        sorted_results = sorted(ocr_results, 
                              key=lambda x: (x['confidence'], -x['bbox'][0][1]), 
                              reverse=True)
        
        # Look for game title patterns
        for result in sorted_results:
            text = result['text'].strip()
            if len(text) > 3 and result['confidence'] > 0.7:
                return text
                
        return sorted_results[0]['text'] if sorted_results else None
```

### Lightweight Vision Model Integration

```python
# vision_models.py - RTX 3050 Optimized
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import gc

class LightweightVisionProcessor:
    def __init__(self):
        # Use smaller CLIP model that fits in 4GB VRAM
        self.model_name = "openai/clip-vit-base-patch32"  # ~600MB
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model with memory optimization
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load model with memory management"""
        try:
            if torch.cuda.is_available():
                # Clear any existing GPU memory
                torch.cuda.empty_cache()
                
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
                self.model.half()  # Use FP16 to save memory
                
        except Exception as e:
            print(f"Failed to load vision model on GPU, falling back to CPU: {e}")
            self.device = 'cpu'
            self.model = CLIPModel.from_pretrained(self.model_name)
    
    def unload_model(self):
        """Free GPU memory when not needed"""
        if self.model is not None:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def classify_game_image(self, image_path, candidate_titles):
        """Classify game image against candidate titles"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Create text prompts for game titles
            text_prompts = [f"a video game box for {title}" for title in candidate_titles]
            text_prompts.append("a video game box cover")
            
            # Process inputs
            inputs = self.processor(
                text=text_prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get best match
            best_match_idx = probs.argmax().item()
            confidence = probs[0][best_match_idx].item()
            
            if best_match_idx < len(candidate_titles) and confidence > 0.3:
                return {
                    'title': candidate_titles[best_match_idx],
                    'confidence': confidence
                }
            
            return None
            
        except Exception as e:
            print(f"Vision classification failed: {e}")
            return None
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## Week 2: Memory-Efficient Image Processing

### Image Segmentation (SAM Alternative)
Since Meta SAM requires >8GB VRAM, we'll use lightweight alternatives:

```python
# lightweight_segmentation.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

class LightweightSegmentation:
    def __init__(self):
        self.methods = ['watershed', 'kmeans', 'contour']
    
    def segment_game_box(self, image_path):
        """Lightweight game box segmentation"""
        image = cv2.imread(image_path)
        
        # Method 1: Contour-based segmentation
        segments = self.contour_segmentation(image)
        
        if not segments:
            # Method 2: K-means clustering
            segments = self.kmeans_segmentation(image)
        
        return segments
    
    def contour_segmentation(self, image):
        """Find game box using contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest rectangular contour (likely game box)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                return [approx]
        
        return []
    
    def kmeans_segmentation(self, image, k=3):
        """Simple K-means segmentation"""
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to image
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        return [segmented_image]
```

## Week 3: Platform Detection Pipeline

### Lightweight Platform Classification

```python
# platform_detection.py - RTX 3050 Optimized
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import json

class PlatformDetector:
    def __init__(self):
        # Use a smaller, fine-tuned model or create custom classifier
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.platform_keywords = {
            'ps5': ['playstation 5', 'ps5', 'sony'],
            'ps4': ['playstation 4', 'ps4', 'sony'],
            'switch': ['nintendo switch', 'switch', 'nintendo'],
            'xbox': ['xbox', 'microsoft', 'xbox one', 'xbox series'],
            'pc': ['pc', 'steam', 'windows', 'computer']
        }
    
    def detect_platform_from_text(self, ocr_results):
        """Detect platform from OCR text"""
        if not ocr_results:
            return None
            
        # Combine all text
        full_text = ' '.join([result['text'].lower() for result in ocr_results])
        
        # Score each platform
        platform_scores = {}
        for platform, keywords in self.platform_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                platform_scores[platform] = score
        
        if platform_scores:
            best_platform = max(platform_scores, key=platform_scores.get)
            confidence = platform_scores[best_platform] / len(self.platform_keywords[best_platform])
            return {
                'platform': best_platform,
                'confidence': min(confidence, 1.0)
            }
        
        return None
    
    def detect_platform_from_colors(self, image_path):
        """Detect platform from dominant colors (simple heuristic)"""
        import cv2
        
        image = cv2.imread(image_path)
        
        # Get dominant colors
        dominant_colors = self.get_dominant_colors(image)
        
        # Platform color associations (simplified)
        color_associations = {
            'ps5': [(255, 255, 255), (0, 0, 0)],  # White/Black
            'ps4': [(0, 100, 200), (0, 0, 0)],    # Blue/Black
            'switch': [(255, 0, 0), (0, 0, 255)], # Red/Blue
            'xbox': [(0, 128, 0), (0, 0, 0)]      # Green/Black
        }
        
        # Simple color matching logic
        for platform, colors in color_associations.items():
            for color in colors:
                if self.color_similarity(dominant_colors[0], color) > 0.8:
                    return {
                        'platform': platform,
                        'confidence': 0.6  # Lower confidence for color-based detection
                    }
        
        return None
    
    def get_dominant_colors(self, image, k=3):
        """Get dominant colors using K-means"""
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers.astype(int)
    
    def color_similarity(self, color1, color2):
        """Calculate color similarity (0-1)"""
        diff = np.abs(np.array(color1) - np.array(color2))
        return 1 - (np.sum(diff) / (3 * 255))
```

## Week 4: Integration & Testing

### Memory Management System

```python
# memory_manager.py - RTX 3050 Specific
import torch
import psutil
import gc
from contextlib import contextmanager

class RTX3050MemoryManager:
    def __init__(self):
        self.max_vram_usage = 3.5 * 1024 * 1024 * 1024  # 3.5GB limit
        self.model_registry = {}
        self.current_models = []
    
    def get_memory_info(self):
        """Get current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0)
            gpu_total = torch.cuda.get_device_properties(0).total_memory
        else:
            gpu_memory = 0
            gpu_total = 0
            
        cpu_memory = psutil.virtual_memory()
        
        return {
            'gpu_used': gpu_memory,
            'gpu_total': gpu_total,
            'gpu_free': gpu_total - gpu_memory,
            'cpu_used': cpu_memory.used,
            'cpu_total': cpu_memory.total,
            'cpu_percent': cpu_memory.percent
        }
    
    @contextmanager
    def managed_model_loading(self, model_name, estimated_size_mb):
        """Context manager for safe model loading"""
        try:
            # Check if we have enough memory
            if not self.can_load_model(estimated_size_mb):
                self.free_least_used_model()
            
            # Load model
            yield
            
            # Register model
            self.register_model(model_name, estimated_size_mb)
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.cleanup_failed_load()
            raise
        finally:
            # Always cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def can_load_model(self, size_mb):
        """Check if model can be loaded"""
        size_bytes = size_mb * 1024 * 1024
        memory_info = self.get_memory_info()
        
        return memory_info['gpu_free'] > size_bytes
    
    def register_model(self, name, size_mb):
        """Register loaded model"""
        self.model_registry[name] = {
            'size_mb': size_mb,
            'last_used': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        if name not in self.current_models:
            self.current_models.append(name)
    
    def free_least_used_model(self):
        """Free the least recently used model"""
        if not self.current_models:
            return
            
        # Simple LRU - remove first model (implement proper LRU if needed)
        model_to_remove = self.current_models.pop(0)
        del self.model_registry[model_to_remove]
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

## Updated Docker Configuration

```dockerfile
# Dockerfile.rtx3050 - Optimized for 4GB VRAM
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3-pip \
    git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with specific versions for RTX 3050
RUN pip install --no-cache-dir \
    torch==2.1.1+cu121 torchvision==0.16.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install lightweight ML packages
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    sentence-transformers==2.2.2 \
    easyocr==1.7.0 \
    opencv-python==4.8.1.78 \
    pillow==10.1.0 \
    scikit-learn==1.3.2

# Set memory limits
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app
```

## Performance Expectations

### RTX 3050 Performance Targets:
- **OCR Processing**: 2-5 seconds per image
- **Vision Classification**: 3-8 seconds per image
- **Platform Detection**: 1-3 seconds per image
- **Memory Usage**: <3.5GB VRAM peak

### Optimization Strategies:
1. **Model Swapping**: Load/unload models as needed
2. **CPU Fallbacks**: Use CPU for less critical tasks
3. **Quantization**: FP16 models where possible
4. **Batch Processing**: Process multiple images efficiently

## Testing Checklist

- [ ] OCR pipeline works with <2GB VRAM usage
- [ ] Vision models load without OOM errors
- [ ] Memory management prevents crashes
- [ ] Performance meets minimum requirements
- [ ] CPU fallbacks work correctly
- [ ] Model swapping functions properly

## Next Phase Preview
Phase 3 will focus on advanced features with continued memory optimization, including marketplace integration and intelligent pricing algorithms that work within hardware constraints.
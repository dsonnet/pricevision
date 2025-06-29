"""
PriceVision Memory Manager
RTX 3050 optimized memory management for AI models
"""

import gc
import psutil
try:
    import torch
except ImportError:
    torch = None
import logging
from typing import Dict, Optional, Any, List, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import threading
import time

from .config import config

logger = logging.getLogger(__name__)

class ModelPriority(Enum):
    """Model loading priority levels"""
    HIGH = 1      # Always keep in memory
    MEDIUM = 2    # Load when needed, unload if memory pressure
    LOW = 3       # Load only when needed, unload immediately

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    model: Any
    size_mb: float
    priority: ModelPriority
    last_used: float
    load_time: float
    usage_count: int

class MemoryManager:
    """RTX 3050 optimized memory manager"""
    
    def __init__(self):
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_loaders: Dict[str, Dict[str, Any]] = {}
        self.max_gpu_memory_mb = config.rtx3050.max_gpu_memory_mb
        self.max_system_memory_mb = config.rtx3050.max_system_memory_mb
        self.memory_threshold = 0.85  # 85% threshold for cleanup
        self._lock = threading.Lock()
        
        # Initialize GPU memory management
        if torch and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
            torch.cuda.empty_cache()
    
    def register_model_loader(self, name: str, loader: Callable, priority: ModelPriority = ModelPriority.MEDIUM):
        """Register a model loader function"""
        self.model_loaders[name] = {
            'loader': loader,
            'priority': priority
        }
        logger.info(f"📝 Registered model loader: {name} (priority: {priority.name})")
    
    async def load_model(self, name: str, force_reload: bool = False) -> Any:
        """Load a model with memory management"""
        with self._lock:
            # Check if model is already loaded
            if name in self.loaded_models and not force_reload:
                model_info = self.loaded_models[name]
                model_info.last_used = time.time()
                model_info.usage_count += 1
                logger.debug(f"🔄 Using cached model: {name}")
                return model_info.model
            
            # Check if loader exists
            if name not in self.model_loaders:
                raise ValueError(f"No loader registered for model: {name}")
            
            # Check memory before loading
            await self._ensure_memory_available()
            
            # Load the model
            start_time = time.time()
            loader_info = self.model_loaders[name]
            
            try:
                logger.info(f"🔄 Loading model: {name}")
                model = loader_info['loader']()
                load_time = time.time() - start_time
                
                # Calculate model size
                model_size = self._calculate_model_size(model)
                
                # Store model info
                self.loaded_models[name] = ModelInfo(
                    name=name,
                    model=model,
                    size_mb=model_size,
                    priority=loader_info['priority'],
                    last_used=time.time(),
                    load_time=load_time,
                    usage_count=1
                )
                
                logger.info(f"✅ Model loaded: {name} ({model_size:.1f}MB, {load_time:.2f}s)")
                return model
                
            except Exception as e:
                logger.error(f"❌ Failed to load model {name}: {e}")
                raise
    
    def unload_model(self, name: str):
        """Unload a specific model"""
        with self._lock:
            if name in self.loaded_models:
                model_info = self.loaded_models[name]
                del self.loaded_models[name]
                del model_info.model
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"🗑️ Unloaded model: {name}")
    
    async def _ensure_memory_available(self):
        """Ensure sufficient memory is available"""
        gpu_usage = self._get_gpu_memory_usage()
        system_usage = self._get_system_memory_usage()
        
        # Check if cleanup is needed
        if (gpu_usage > self.memory_threshold or 
            system_usage > self.memory_threshold):
            await self._cleanup_memory()
    
    async def _cleanup_memory(self):
        """Clean up memory by unloading low-priority models"""
        logger.info("🧹 Starting memory cleanup")
        
        # Sort models by priority and last used time
        models_to_unload = []
        for name, model_info in self.loaded_models.items():
            if model_info.priority != ModelPriority.HIGH:
                models_to_unload.append((name, model_info))
        
        # Sort by priority (low first) and last used time (oldest first)
        models_to_unload.sort(key=lambda x: (x[1].priority.value, x[1].last_used))
        
        # Unload models until memory usage is acceptable
        for name, model_info in models_to_unload:
            self.unload_model(name)
            
            # Check if we have enough memory now
            gpu_usage = self._get_gpu_memory_usage()
            system_usage = self._get_system_memory_usage()
            
            if (gpu_usage < self.memory_threshold and 
                system_usage < self.memory_threshold):
                break
        
        logger.info("✅ Memory cleanup completed")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        if not torch or not torch.cuda.is_available():
            return 0.0
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            return allocated / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def _get_system_memory_usage(self) -> float:
        """Get current system memory usage percentage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0
    
    def _calculate_model_size(self, model) -> float:
        """Calculate model size in MB"""
        try:
            if hasattr(model, 'parameters'):
                # PyTorch model
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / 1024**2
            else:
                # Estimate based on object size
                import sys
                return sys.getsizeof(model) / 1024**2
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        gpu_usage = self._get_gpu_memory_usage()
        system_usage = self._get_system_memory_usage()
        
        loaded_models = {
            name: {
                'size_mb': info.size_mb,
                'priority': info.priority.name,
                'last_used': info.last_used,
                'usage_count': info.usage_count
            }
            for name, info in self.loaded_models.items()
        }
        
        return {
            'gpu_memory_usage': gpu_usage,
            'system_memory_usage': system_usage,
            'loaded_models': loaded_models,
            'total_model_size_mb': sum(info.size_mb for info in self.loaded_models.values())
        }
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        if not torch or not torch.cuda.is_available():
            return {
                "available": False,
                "device_count": 0,
                "current_device": None,
                "memory": {}
            }
        
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Memory information
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            total = torch.cuda.get_device_properties(current_device).total_memory / 1024**2  # MB
            
            return {
                "available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "memory": {
                    "allocated_mb": round(allocated, 2),
                    "cached_mb": round(cached, 2),
                    "total_mb": round(total, 2),
                    "free_mb": round(total - allocated, 2),
                    "usage_percent": round((allocated / total) * 100, 2) if total > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage information"""
        gpu_usage = self._get_gpu_memory_usage()
        system_usage = self._get_system_memory_usage()
        
        # System memory details
        try:
            memory = psutil.virtual_memory()
            system_memory = {
                "total_mb": round(memory.total / 1024**2, 2),
                "available_mb": round(memory.available / 1024**2, 2),
                "used_mb": round(memory.used / 1024**2, 2),
                "usage_percent": round(memory.percent, 2)
            }
        except Exception:
            system_memory = {"error": "Unable to get system memory info"}
        
        # GPU memory details
        gpu_status = await self.get_gpu_status()
        gpu_memory = gpu_status.get("memory", {})
        
        # Model memory usage
        model_memory = {
            "loaded_models_count": len(self.loaded_models),
            "total_model_size_mb": round(sum(info.size_mb for info in self.loaded_models.values()), 2),
            "models": {
                name: {
                    "size_mb": round(info.size_mb, 2),
                    "priority": info.priority.name,
                    "usage_count": info.usage_count
                }
                for name, info in self.loaded_models.items()
            }
        }
        
        return {
            "system": system_memory,
            "gpu": gpu_memory,
            "models": model_memory,
            "thresholds": {
                "memory_threshold": self.memory_threshold,
                "max_gpu_memory_mb": self.max_gpu_memory_mb,
                "max_system_memory_mb": self.max_system_memory_mb
            }
        }
    
    async def cleanup_all(self):
        """Clean up all loaded models and free memory"""
        logger.info("🧹 Starting complete memory cleanup")
        
        with self._lock:
            # Unload all models except HIGH priority ones
            models_to_unload = [
                name for name, info in self.loaded_models.items()
                if info.priority != ModelPriority.HIGH
            ]
            
            for model_name in models_to_unload:
                self.unload_model(model_name)
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✅ Complete memory cleanup finished")
    
    @asynccontextmanager
    async def model_context(self, model_name: str):
        """Async context manager for temporary model loading"""
        model = None
        try:
            model = await self.load_model(model_name)
            yield model
        finally:
            # For low priority models, unload immediately
            if (model_name in self.loaded_models and
                self.loaded_models[model_name].priority == ModelPriority.LOW):
                self.unload_model(model_name)
    
    @asynccontextmanager
    async def load_model_context(self, model_name: str, priority: ModelPriority = ModelPriority.MEDIUM):
        """Async context manager for loading models with automatic cleanup"""
        model = None
        try:
            # Load the model with specified priority
            model = await self.load_model(model_name)
            yield model
        finally:
            # For low priority models, unload immediately
            if (model_name in self.loaded_models and
                self.loaded_models[model_name].priority == ModelPriority.LOW):
                self.unload_model(model_name)
            
            # Check if cleanup is needed after model usage
            await self.cleanup_if_needed()
    
    async def cleanup_if_needed(self):
        """Check memory usage and cleanup if needed"""
        try:
            gpu_usage = self._get_gpu_memory_usage()
            system_usage = self._get_system_memory_usage()
            
            # Cleanup if GPU memory usage is above 80% or system memory above 85%
            if gpu_usage > 80.0 or system_usage > 85.0:
                logger.warning(f"High memory usage detected - GPU: {gpu_usage:.1f}%, System: {system_usage:.1f}%")
                
                # Unload LOW priority models first
                with self._lock:
                    low_priority_models = [
                        name for name, info in self.loaded_models.items()
                        if info.priority == ModelPriority.LOW
                    ]
                    
                    for model_name in low_priority_models:
                        self.unload_model(model_name)
                        logger.info(f"Unloaded low priority model: {model_name}")
                
                # Force garbage collection
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Check again and unload MEDIUM priority if still high
                gpu_usage = self._get_gpu_memory_usage()
                if gpu_usage > 85.0:
                    with self._lock:
                        medium_priority_models = [
                            name for name, info in self.loaded_models.items()
                            if info.priority == ModelPriority.MEDIUM
                        ]
                        
                        for model_name in medium_priority_models[:2]:  # Unload up to 2 medium priority models
                            self.unload_model(model_name)
                            logger.info(f"Unloaded medium priority model: {model_name}")
                
                logger.info("Memory cleanup completed")
                
        except Exception as e:
            logger.error(f"Cleanup check failed: {e}")

class GPUMemoryMonitor:
    """Monitor GPU memory usage for RTX 3050"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """Add callback for memory alerts"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 GPU memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("⏹️ GPU memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring:
            try:
                if torch and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    cached = torch.cuda.memory_reserved() / 1024**2
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    
                    usage_percent = allocated / total if total > 0 else 0
                    
                    memory_info = {
                        'allocated_mb': allocated,
                        'cached_mb': cached,
                        'total_mb': total,
                        'usage_percent': usage_percent,
                        'timestamp': time.time()
                    }
                    
                    # Alert if usage is high
                    if usage_percent > 0.9:  # 90% threshold
                        logger.warning(f"⚠️ High GPU memory usage: {usage_percent:.1%}")
                        for callback in self.callbacks:
                            callback(memory_info)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)

# Global instances
memory_manager = MemoryManager()
gpu_monitor = GPUMemoryMonitor()

# Convenience functions
async def load_model(name: str, force_reload: bool = False):
    """Load model with memory management"""
    return await memory_manager.load_model(name, force_reload)

def unload_model(name: str):
    """Unload model"""
    memory_manager.unload_model(name)

def register_model_loader(name: str, loader: Callable, priority: ModelPriority = ModelPriority.MEDIUM):
    """Register model loader"""
    memory_manager.register_model_loader(name, loader, priority)

def get_memory_stats():
    """Get memory statistics"""
    return memory_manager.get_memory_stats()

def model_context(model_name: str):
    """Async context manager for model loading"""
    return memory_manager.model_context(model_name)

def start_memory_monitoring():
    """Start GPU memory monitoring"""
    gpu_monitor.start_monitoring()

def stop_memory_monitoring():
    """Stop GPU memory monitoring"""
    gpu_monitor.stop_monitoring()

async def initialize_memory_manager():
    """Initialize the memory manager system"""
    logger.info("🚀 Initializing RTX 3050 Memory Manager")
    
    # Start GPU monitoring if available
    if torch and torch.cuda.is_available():
        start_memory_monitoring()
        logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("⚠️ No GPU detected, running in CPU-only mode")
    
    # Log memory configuration
    logger.info(f"🔧 Max GPU Memory: {memory_manager.max_gpu_memory_mb}MB")
    logger.info(f"🔧 Max System Memory: {memory_manager.max_system_memory_mb}MB")
    logger.info(f"🔧 Memory Threshold: {memory_manager.memory_threshold:.0%}")
    
    logger.info("✅ Memory Manager initialized successfully")
"""
PriceVision Configuration Module
RTX 3050 Optimized Settings
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path

class RTX3050Config(BaseSettings):
    """RTX 3050 specific GPU configuration"""
    max_gpu_memory_gb: float = Field(default=3.5, description="Maximum GPU memory usage in GB")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    model_cache_size: int = Field(default=2, description="Number of models to keep in memory")
    enable_model_swapping: bool = Field(default=True, description="Enable dynamic model loading/unloading")
    pytorch_cuda_alloc_conf: str = Field(default="max_split_size_mb:512", description="PyTorch CUDA memory allocation")
    cuda_visible_devices: str = Field(default="0", description="CUDA visible devices")
    omp_num_threads: int = Field(default=4, description="OpenMP thread count")
    tokenizers_parallelism: bool = Field(default=False, description="Tokenizers parallelism")

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_database: str = Field(default="pricevision", env="MYSQL_DATABASE")
    mysql_user: str = Field(default="pricevision_user", env="MYSQL_USER")
    mysql_password: str = Field(default="pricevision_pass", env="MYSQL_PASSWORD")
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8001, env="CHROMADB_PORT")
    
    @property
    def mysql_url(self) -> str:
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    @property
    def chromadb_url(self) -> str:
        return f"http://{self.chromadb_host}:{self.chromadb_port}"

class AIConfig(BaseSettings):
    """AI and ML model configuration"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Lightweight models for RTX 3050
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Lightweight embedding model")
    classification_model: str = Field(default="distilbert-base-uncased-finetuned-sst-2-english", description="Text classification model")
    
    # OCR Configuration (CPU-based for RTX 3050)
    ocr_languages: List[str] = Field(default=["en", "fr"], description="OCR supported languages")
    ocr_gpu_enabled: bool = Field(default=False, description="Use GPU for OCR (disabled for RTX 3050)")
    
    # Image processing
    max_image_size_mb: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    supported_image_formats: List[str] = Field(default=["jpg", "jpeg", "png", "webp"])

class RasaConfig(BaseSettings):
    """Rasa chatbot configuration"""
    model_path: str = Field(default="./models", env="RASA_MODEL_PATH")
    actions_endpoint: str = Field(default="http://localhost:5055/webhook", env="RASA_ACTIONS_ENDPOINT")
    core_endpoint: str = Field(default="http://localhost:5005", description="Rasa Core endpoint")
    
class APIConfig(BaseSettings):
    """FastAPI configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")  # Single worker for RTX 3050
    debug: bool = Field(default=True, env="DEBUG")
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])

class MarketplaceConfig(BaseSettings):
    """Marketplace API configuration"""
    ebay_app_id: Optional[str] = Field(default=None, env="EBAY_APP_ID")
    ebay_cert_id: Optional[str] = Field(default=None, env="EBAY_CERT_ID")
    ebay_dev_id: Optional[str] = Field(default=None, env="EBAY_DEV_ID")
    ebay_user_token: Optional[str] = Field(default=None, env="EBAY_USER_TOKEN")
    
    rakuten_application_id: Optional[str] = Field(default=None, env="RAKUTEN_APPLICATION_ID")
    rakuten_affiliate_id: Optional[str] = Field(default=None, env="RAKUTEN_AFFILIATE_ID")

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="logs/pricevision.log", env="LOG_FILE")

class PriceVisionConfig(BaseSettings):
    """Main PriceVision configuration"""
    
    # Sub-configurations
    rtx3050: RTX3050Config = RTX3050Config()
    database: DatabaseConfig = DatabaseConfig()
    ai: AIConfig = AIConfig()
    rasa: RasaConfig = RasaConfig()
    api: APIConfig = APIConfig()
    marketplace: MarketplaceConfig = MarketplaceConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # General settings
    secret_key: str = Field(default="your_secret_key_here", env="SECRET_KEY")
    jwt_secret: str = Field(default="your_jwt_secret_here", env="JWT_SECRET")
    development_mode: bool = Field(default=True, env="DEVELOPMENT_MODE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global configuration instance
config = PriceVisionConfig()

def setup_gpu_environment():
    """Setup RTX 3050 specific environment variables"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.rtx3050.pytorch_cuda_alloc_conf
    os.environ["CUDA_VISIBLE_DEVICES"] = config.rtx3050.cuda_visible_devices
    os.environ["OMP_NUM_THREADS"] = str(config.rtx3050.omp_num_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = str(config.rtx3050.tokenizers_parallelism).lower()

def get_model_device():
    """Get the appropriate device for model loading"""
    import torch
    if torch.cuda.is_available() and config.rtx3050.cuda_visible_devices != "-1":
        return "cuda:0"
    return "cpu"

def check_gpu_memory():
    """Check available GPU memory for RTX 3050"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            
            return {
                "total_gb": total_memory / (1024**3),
                "allocated_gb": allocated_memory / (1024**3),
                "free_gb": free_memory / (1024**3),
                "utilization_percent": (allocated_memory / total_memory) * 100
            }
    except Exception:
        pass
    
    return {
        "total_gb": 0,
        "allocated_gb": 0,
        "free_gb": 0,
        "utilization_percent": 0
    }

# Initialize GPU environment on import
setup_gpu_environment()
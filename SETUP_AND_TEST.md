# PriceVision Setup and Testing Guide
## RTX 3050 Optimized Implementation

This guide walks you through setting up and testing the complete PriceVision system optimized for RTX 3050 (4GB VRAM).

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-rtx3050.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=pricevision
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=pricevision

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# AI Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# RTX 3050 Optimization
RTX3050_MAX_GPU_MEMORY_GB=3.5
RTX3050_ENABLE_MODEL_SWAPPING=true
RTX3050_MAX_CONCURRENT_REQUESTS=2
```

### 3. Database Setup

#### Option A: Docker (Recommended)
```bash
# Start databases with Docker
docker-compose -f docker-compose.dev.yml up -d

# Wait for databases to be ready
sleep 30

# Initialize database schema
docker exec -i pricevision-mysql mysql -u pricevision -p pricevision < database/init/01-create-tables.sql
docker exec -i pricevision-mysql mysql -u pricevision -p pricevision < database/init/02-populate-data.sql
```

#### Option B: Local Installation
```bash
# Install MySQL and Redis locally
# Then run the SQL scripts manually
mysql -u root -p < database/init/01-create-tables.sql
mysql -u root -p < database/init/02-populate-data.sql
```

### 4. System Verification

```bash
# Verify RTX 3050 setup
python scripts/verify_rtx3050.sh

# Test system components
python scripts/test_rtx3050_setup.py

# Run integration tests
python scripts/test_integration.py
```

### 5. Start the System

```bash
# Option A: Using the startup script (Recommended)
python scripts/start_system.py

# Option B: Direct app startup
python src/app.py

# Option C: Using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 🧪 Testing the System

### API Endpoints

Once the system is running, test these endpoints:

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the price of Pokemon cards?"}'
```

#### Photo Analysis
```bash
curl -X POST http://localhost:8000/analyze-photo \
  -F "file=@path/to/game/image.jpg"
```

#### Price Inquiry
```bash
curl -X POST http://localhost:8000/price-inquiry \
  -H "Content-Type: application/json" \
  -d '{"item_name": "Pokemon Charizard Card", "condition": "mint"}'
```

#### System Status
```bash
curl http://localhost:8000/system/status
```

### Web Interface

Open your browser and navigate to:
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
```

#### 2. Memory Issues
```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check system memory
free -h

# Monitor Python memory usage
python -c "import psutil; print(f'RAM Usage: {psutil.virtual_memory().percent}%')"
```

#### 3. Database Connection Issues
```bash
# Test MySQL connection
mysql -h localhost -u pricevision -p -e "SELECT 1;"

# Test Redis connection
redis-cli ping

# Check Docker containers
docker ps
docker logs pricevision-mysql
docker logs pricevision-redis
```

#### 4. Import Errors
```bash
# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check installed packages
pip list | grep -E "(rasa|fastapi|sqlalchemy|torch)"

# Reinstall requirements
pip install -r requirements-rtx3050.txt --force-reinstall
```

### Performance Optimization

#### RTX 3050 Specific Settings

1. **Memory Management**
   - Max GPU memory: 3.5GB (leaves 0.5GB buffer)
   - Model swapping enabled
   - Single worker FastAPI configuration

2. **Model Optimization**
   - Lightweight models: DistilBERT, MiniLM
   - CPU-based OCR processing
   - Reduced batch sizes and epochs

3. **Caching Strategy**
   - Redis for model caching
   - ChromaDB for vector storage
   - Aggressive cleanup policies

## 📊 Monitoring

### System Metrics

```bash
# GPU utilization
nvidia-smi -l 1

# System resources
htop

# Application logs
tail -f logs/pricevision.log

# Database performance
docker exec pricevision-mysql mysqladmin -u pricevision -p processlist
```

### Performance Benchmarks

Expected performance on RTX 3050:

- **Chat Response**: < 2 seconds
- **Photo Analysis**: < 5 seconds
- **Price Lookup**: < 1 second
- **Memory Usage**: < 3.5GB GPU, < 4GB RAM
- **Concurrent Users**: 2-3 simultaneous

## 🔄 Development Workflow

### Making Changes

1. **Code Changes**
   ```bash
   # Edit source files
   # Test changes
   python scripts/test_integration.py
   
   # Restart system
   python scripts/start_system.py
   ```

2. **Database Changes**
   ```bash
   # Update schema files
   # Apply changes
   mysql -u pricevision -p pricevision < your_migration.sql
   ```

3. **Configuration Changes**
   ```bash
   # Update .env file
   # Restart system for changes to take effect
   ```

### Adding New Features

1. Follow the Phase implementation structure
2. Maintain RTX 3050 memory constraints
3. Add appropriate tests
4. Update documentation

## 📚 Architecture Overview

### Core Components

1. **FastAPI Application** (`src/main.py`)
   - RESTful API endpoints
   - RTX 3050 optimized configuration
   - Async request handling

2. **Memory Manager** (`src/memory_manager.py`)
   - GPU memory monitoring
   - Dynamic model loading/unloading
   - Priority-based cleanup

3. **Database Layer** (`src/database.py`)
   - MySQL for structured data
   - Redis for caching
   - ChromaDB for vector storage

4. **AI Components**
   - **RAG System** (`src/rag_system.py`): LlamaIndex integration
   - **Photo Analysis** (`src/photo_analysis.py`): EasyOCR processing
   - **Rasa Chatbot** (`rasa/`): Conversational AI

5. **Configuration** (`src/config.py`)
   - Pydantic-based settings
   - Environment variable management
   - RTX 3050 specific optimizations

### Data Flow

1. **User Request** → FastAPI endpoint
2. **Memory Check** → Ensure GPU memory available
3. **Component Loading** → Load required AI models
4. **Processing** → Execute AI operations
5. **Cleanup** → Unload models if needed
6. **Response** → Return results to user

## 🎯 Next Steps

After successful Phase 1 testing:

1. **Phase 2**: Advanced AI Features
2. **Phase 3**: Web Interface Development
3. **Phase 4**: Mobile Integration
4. **Phase 5**: Production Deployment

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Run diagnostic scripts in `scripts/` directory
4. Check system requirements and dependencies

---

**Note**: This system is optimized for RTX 3050 with 4GB VRAM. Performance may vary on different hardware configurations.
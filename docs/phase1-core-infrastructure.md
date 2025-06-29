# Phase 1: Core Infrastructure Setup (4-6 weeks)

## Overview
This phase establishes the foundation for the PriceVision chatbot system, including environment setup, database enhancements, Rasa framework configuration, and LlamaIndex RAG system implementation.

## Objectives
- Set up development environment with GPU support
- Enhance existing MySQL schema for AI integration
- Configure Rasa conversational AI framework
- Implement LlamaIndex RAG system for intelligent data retrieval

## Week 1-2: Environment and Database Setup

### System Requirements Verification
**Your Current System:**
- **OS**: Linux 5.15 ✅
- **GPU**: NVIDIA GeForce RTX 3050 (4096MiB VRAM) ✅
- **Development Team**: Python/AI expertise available ✅
- **GPU Resources**: Available for project ✅

### Infrastructure Setup
**Deliverables:**
- Docker development environment optimized for RTX 3050
- GPU-enabled containers with 4GB VRAM constraints
- Development database instances
- RTX 3050-specific configuration files

**Tasks:**
1. **RTX 3050 Environment Verification**
   ```bash
   # verify_rtx3050.sh - System Requirements Check
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
   ```

2. **Docker Environment Configuration (RTX 3050 Optimized)**
   ```dockerfile
   # Dockerfile.dev - RTX 3050 Optimized
   FROM nvidia/cuda:12.1-devel-ubuntu20.04
   
   # Set environment variables for RTX 3050
   ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ENV CUDA_VISIBLE_DEVICES=0
   ENV OMP_NUM_THREADS=4
   ENV TOKENIZERS_PARALLELISM=false
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       python3.9 python3-pip python3-dev \
       git curl wget unzip \
       libgl1-mesa-glx libglib2.0-0 \
       tesseract-ocr tesseract-ocr-fra \
       && rm -rf /var/lib/apt/lists/*
   
   # Install PyTorch with CUDA 12.1 support
   RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
       --index-url https://download.pytorch.org/whl/cu121
   
   # Install RTX 3050 optimized packages
   RUN pip install \
       transformers==4.35.0 \
       sentence-transformers==2.2.2 \
       easyocr==1.7.0 \
       opencv-python==4.8.1.78 \
       pillow==10.0.1
   
   WORKDIR /app
   
   # Copy requirements and install
   COPY requirements-rtx3050.txt .
   RUN pip install -r requirements-rtx3050.txt
   
   # Health check for GPU
   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
       CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
   ```

3. **RTX 3050 GPU Environment Verification**
   ```python
   # test_rtx3050_setup.py
   import torch
   import subprocess
   import psutil
   from transformers import pipeline
   import easyocr
   
   def verify_rtx3050_setup():
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
           reader = easyocr.Reader(['en', 'fr'], gpu=False)  # CPU for RTX 3050
           print("✅ EasyOCR initialized (CPU mode)")
       except Exception as e:
           print(f"❌ EasyOCR failed: {e}")
       
       # Test lightweight transformer
       try:
           # Use small model for RTX 3050
           classifier = pipeline("text-classification",
                               model="distilbert-base-uncased-finetuned-sst-2-english",
                               device=0 if cuda_available else -1)
           result = classifier("This is a test")
           print("✅ Transformers pipeline working")
       except Exception as e:
           print(f"❌ Transformers failed: {e}")
       
       print("\n🎯 RTX 3050 Optimization Settings:")
       print(f"   - Max GPU Memory Usage: 3.5GB (87.5% of 4GB)")
       print(f"   - Concurrent Requests: 5 maximum")
       print(f"   - Model Loading: Dynamic (no preloading)")
       print(f"   - OCR Processing: CPU-based")
       print(f"   - Embedding Models: CPU-based")
       
   if __name__ == "__main__":
       verify_rtx3050_setup()
   ```

3. **Development Database Setup**
   - MySQL instance for main data
   - ChromaDB instance for vector storage
   - Redis instance for caching

### Database Schema Enhancement

**New Tables Creation:**
```sql
-- Platforms lookup table
CREATE TABLE platforms (
    platform_id INT PRIMARY KEY AUTO_INCREMENT,
    platform_name VARCHAR(50) NOT NULL,
    platform_short VARCHAR(10) NOT NULL,
    UNIQUE KEY (platform_name)
);

-- Genres lookup table  
CREATE TABLE genres (
    genre_id INT PRIMARY KEY AUTO_INCREMENT,
    genre_name VARCHAR(50) NOT NULL,
    UNIQUE KEY (genre_name)
);

-- Marketplace pricing data
CREATE TABLE marketplace_prices (
    id INT PRIMARY KEY AUTO_INCREMENT,
    products_model VARCHAR(15) NOT NULL,
    source ENUM('ebay', 'rakuten_fr', 'vinted', 'internal_estimate'),
    listing_title VARCHAR(255),
    price DECIMAL(8,2),
    condition_grade ENUM('mint', 'near_mint', 'good', 'fair', 'poor'),
    listing_date DATE,
    listing_url VARCHAR(500),
    is_sold BOOLEAN DEFAULT FALSE,
    currency VARCHAR(3) DEFAULT 'EUR',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (products_model) REFERENCES products(products_model),
    INDEX idx_model_condition (products_model, condition_grade),
    INDEX idx_date_source (listing_date, source)
);

-- Photo analysis cache
CREATE TABLE photo_analysis_cache (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_hash VARCHAR(64) UNIQUE,
    detected_model VARCHAR(15),
    detected_platform_id INT,
    detected_condition VARCHAR(50),
    detected_gtin VARCHAR(13),
    confidence_score DECIMAL(3,2),
    analysis_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detected_platform_id) REFERENCES platforms(platform_id),
    INDEX idx_hash (image_hash),
    INDEX idx_model (detected_model)
);

-- Document embeddings for RAG
CREATE TABLE ai_document_embeddings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    document_id VARCHAR(100) UNIQUE,
    document_type ENUM('transaction', 'product', 'marketplace_listing'),
    source_id VARCHAR(50),
    content TEXT,
    embedding_vector JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_doc_type (document_type),
    INDEX idx_source (source_id)
);

-- Chatbot conversation history
CREATE TABLE chatbot_conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(64),
    user_query TEXT,
    query_type ENUM('text', 'photo', 'hybrid'),
    detected_intent VARCHAR(50),
    extracted_entities JSON,
    products_model VARCHAR(15),
    estimated_price DECIMAL(8,2),
    confidence_score DECIMAL(3,2),
    response_data JSON,
    processing_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (products_model) REFERENCES products(products_model),
    INDEX idx_session (session_id),
    INDEX idx_model (products_model)
);
```

**Data Migration Scripts:**
```sql
-- Populate platforms table from existing data
INSERT INTO platforms (platform_name, platform_short) 
SELECT DISTINCT platform_name, SUBSTRING(platform_name, 1, 3) 
FROM existing_platform_data;

-- Populate genres table
INSERT INTO genres (genre_name)
SELECT DISTINCT genre_name 
FROM existing_genre_data;
```

### Core Dependencies Installation

**Requirements.txt:**
```txt
# Core AI Framework
rasa==3.6.13
rasa-sdk==3.6.1

# LlamaIndex and RAG
llama-index==0.9.30
llama-index-vector-stores-chroma==0.1.3
llama-index-embeddings-huggingface==0.1.4
chromadb==0.4.18

# Computer Vision (will be used in Phase 2)
segment-anything==1.0
surya-ocr==0.4.14
easyocr==1.7.0
transformers==4.36.2
torch==2.1.1
torchvision==0.16.1

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.2

# Database
pymysql==1.1.0
sqlalchemy==2.0.23
redis==5.0.1

# Utilities
pillow==10.1.0
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.1.4
requests==2.31.0
python-multipart==0.0.6
```

## Week 3-4: Rasa Framework Setup

### Rasa Configuration

**config.yml:**
```yaml
language: en

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
    constrain_similarities: true

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: UnexpecTEDIntentPolicy
    max_history: 5
    epochs: 100
  - name: TEDPolicy
    max_history: 5
    epochs: 100
    constrain_similarities: true
```

**domain.yml:**
```yaml
version: '3.1'

intents:
  - greet
  - goodbye
  - price_inquiry_text
  - price_inquiry_photo
  - condition_inquiry
  - platform_specific
  - market_comparison
  - historical_trend
  - affirm
  - deny

entities:
  - game_title
  - platform
  - condition
  - price_range
  - time_period

slots:
  game_title:
    type: text
    influence_conversation: true
  platform:
    type: categorical
    values:
      - ps5
      - ps4
      - switch
      - xbox
      - pc
    influence_conversation: true
  condition:
    type: categorical
    values:
      - mint
      - near_mint
      - good
      - fair
      - poor
    influence_conversation: true

responses:
  utter_greet:
  - text: "Hello! I can help you price second-hand games. You can ask me about specific games or upload a photo of the game box."

  utter_ask_game_title:
  - text: "What game are you looking to price?"

  utter_ask_platform:
  - text: "What platform is this game for? (PS5, PS4, Nintendo Switch, Xbox, PC)"

  utter_ask_condition:
  - text: "What condition is the game in? (Mint, Near Mint, Good, Fair, Poor)"

  utter_price_result:
  - text: "Based on current market data, {game_title} for {platform} in {condition} condition is estimated at €{price} (±{range}). Confidence: {confidence}%"

  utter_ask_photo_upload:
  - text: "Please upload a clear photo of the game box (front cover preferred) and I'll identify it and provide pricing."

  utter_goodbye:
  - text: "Goodbye! Feel free to ask me about game pricing anytime."

actions:
  - action_process_price_query
  - action_analyze_photo
  - action_get_market_data
  - action_get_price_history

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: true
```

### Training Data Creation

**nlu.yml:**
```yaml
version: "3.1"

nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - moin
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon

- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later

- intent: price_inquiry_text
  examples: |
    - What's the price of [Mario Kart 8 Deluxe](game_title) for [Switch](platform)?
    - How much is [God of War](game_title) on [PS5](platform)?
    - Price for [Zelda Breath of the Wild](game_title)
    - What's [Call of Duty Modern Warfare](game_title) worth?
    - Can you price [FIFA 24](game_title) for [PS4](platform)?
    - How much for [Elden Ring](game_title) on [PC](platform)?
    - What's the value of [Spider-Man Miles Morales](game_title)?
    - Price check on [Cyberpunk 2077](game_title)

- intent: condition_inquiry
  examples: |
    - What's this worth in [mint](condition) condition?
    - How much for [good](condition) condition?
    - Price in [near mint](condition) state?
    - Value if it's [fair](condition) condition?
    - What if it's [poor](condition) condition?

- intent: platform_specific
  examples: |
    - How much for [PS5](platform) games?
    - What are [Switch](platform) games worth?
    - [Xbox](platform) game prices?
    - [PC](platform) game values?

- intent: market_comparison
  examples: |
    - Compare prices across platforms
    - Show me different marketplace prices
    - What's the price range?
    - Compare with eBay prices
    - Market analysis for this game

- intent: historical_trend
  examples: |
    - Show me price history for this game
    - How has the price changed?
    - Price trend over time
    - Historical pricing data
    - Has this game increased in value?
```

### Custom Actions Framework

**actions/actions.py:**
```python
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import mysql.connector
from mysql.connector import pooling
import logging

class DatabaseConnection:
    def __init__(self):
        self.config = {
            'user': 'your_username',
            'password': 'your_password', 
            'host': 'localhost',
            'database': 'your_database',
            'pool_name': 'mypool',
            'pool_size': 5,
            'pool_reset_session': True
        }
        self.pool = mysql.connector.pooling.MySQLConnectionPool(**self.config)
    
    def get_connection(self):
        return self.pool.get_connection()

db = DatabaseConnection()

class ActionProcessPriceQuery(Action):
    def name(self) -> Text:
        return "action_process_price_query"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        game_title = tracker.get_slot("game_title")
        platform = tracker.get_slot("platform")
        condition = tracker.get_slot("condition") or "good"
        
        if not game_title:
            dispatcher.utter_message(template="utter_ask_game_title")
            return []
        
        if not platform:
            dispatcher.utter_message(template="utter_ask_platform")
            return []
        
        try:
            # Basic price query from existing data
            price_data = self.get_price_from_database(game_title, platform, condition)
            
            if price_data:
                dispatcher.utter_message(
                    template="utter_price_result",
                    game_title=game_title,
                    platform=platform,
                    condition=condition,
                    price=price_data['price'],
                    range=price_data['range'],
                    confidence=price_data['confidence']
                )
            else:
                dispatcher.utter_message(
                    text=f"Sorry, I couldn't find pricing data for {game_title} on {platform}. Please try a different game or check the spelling."
                )
                
        except Exception as e:
            logging.error(f"Error in price query: {str(e)}")
            dispatcher.utter_message(
                text="Sorry, I encountered an error while looking up the price. Please try again."
            )
        
        return []
    
    def get_price_from_database(self, game_title: str, platform: str, condition: str):
        """Basic price lookup from existing database"""
        try:
            connection = db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Search for similar game titles in products
            query = """
            SELECT 
                p.products_model,
                pd.products_name,
                p.products_used_price,
                p.products_price,
                AVG(s.items_sold_price_stock) as avg_sold_price
            FROM products p
            JOIN products_description pd ON p.products_id = pd.products_id
            LEFT JOIN netshop_items_sold s ON p.products_model = s.items_sold_model
            WHERE pd.products_name LIKE %s
            AND p.products_platform = (SELECT platform_id FROM platforms WHERE platform_name LIKE %s)
            GROUP BY p.products_model, pd.products_name, p.products_used_price, p.products_price
            LIMIT 1
            """
            
            cursor.execute(query, (f"%{game_title}%", f"%{platform}%"))
            result = cursor.fetchone()
            
            if result:
                base_price = result['products_used_price'] or result['avg_sold_price'] or result['products_price']
                
                # Simple condition adjustment
                condition_multipliers = {
                    'mint': 1.1,
                    'near_mint': 1.0,
                    'good': 0.9,
                    'fair': 0.7,
                    'poor': 0.5
                }
                
                adjusted_price = base_price * condition_multipliers.get(condition, 0.9)
                price_range = adjusted_price * 0.15  # ±15% range
                
                return {
                    'price': round(adjusted_price, 2),
                    'range': round(price_range, 2),
                    'confidence': 75  # Basic confidence score
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
```

## Week 5-6: LlamaIndex RAG System

### Data Ingestion Pipeline

**rag_system/data_ingestion.py:**
```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import chromadb
import mysql.connector
from typing import List, Dict
import json
from datetime import datetime, timedelta

class GamePricingRAG:
    def __init__(self, mysql_config: Dict, chroma_path: str = "./chroma_db"):
        self.mysql_config = mysql_config
        self.chroma_path = chroma_path
        self.setup_components()
    
    def setup_components(self):
        """Initialize LlamaIndex components"""
        # Setup embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        # Setup ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.create_collection("game_pricing")
        
        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize index (will be populated later)
        self.index = None
        self.query_engine = None
    
    def ingest_transaction_history(self):
        """Ingest historical sales data"""
        connection = mysql.connector.connect(**self.mysql_config)
        cursor = connection.cursor(dictionary=True)
        
        # Get transaction history
        query = """
        SELECT 
            CONCAT('SALE_', s.items_sold_id) as doc_id,
            s.items_sold_model,
            pd.products_name,
            s.items_sold_condition_grade,
            s.items_sold_price_stock,
            s.items_sold_price_gross,
            s.items_sold_date_in,
            plat.platform_name
        FROM netshop_items_sold s
        JOIN products p ON s.items_sold_model = p.products_model
        JOIN products_description pd ON p.products_id = pd.products_id
        LEFT JOIN platforms plat ON p.products_platform = plat.platform_id
        WHERE s.items_sold_condition_grade IS NOT NULL
        AND STR_TO_DATE(s.items_sold_date_in, '%Y%m%d') >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        documents = []
        for row in results:
            profit = row['items_sold_price_stock'] - row['items_sold_price_gross']
            content = f"""
            TRANSACTION RECORD:
            Game: {row['products_name']}
            Model: {row['items_sold_model']}
            Platform: {row['platform_name']}
            Condition: {row['items_sold_condition_grade']}
            Sale Price: €{row['items_sold_price_stock']}
            Cost: €{row['items_sold_price_gross']}
            Profit: €{profit}
            Date: {row['items_sold_date_in']}
            """
            
            metadata = {
                "doc_type": "transaction",
                "model": row['items_sold_model'],
                "platform": row['platform_name'],
                "condition": row['items_sold_condition_grade'],
                "price": float(row['items_sold_price_stock']),
                "date": row['items_sold_date_in']
            }
            
            doc = Document(
                text=content.strip(),
                metadata=metadata,
                doc_id=row['doc_id']
            )
            documents.append(doc)
        
        cursor.close()
        connection.close()
        
        return documents
    
    def ingest_product_catalog(self):
        """Ingest current product catalog"""
        connection = mysql.connector.connect(**self.mysql_config)
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT 
            CONCAT('PRODUCT_', p.products_model) as doc_id,
            p.products_model,
            pd.products_name,
            plat.platform_name,
            g.genre_name,
            p.products_price,
            p.products_used_price,
            p.products_margin,
            p.products_quantity,
            p.products_gtin
        FROM products p
        JOIN products_description pd ON p.products_id = pd.products_id
        LEFT JOIN platforms plat ON p.products_platform = plat.platform_id
        LEFT JOIN genres g ON p.products_genre = g.genre_id
        WHERE p.products_status = 1
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        documents = []
        for row in results:
            content = f"""
            PRODUCT CATALOG:
            Game: {row['products_name']}
            Model: {row['products_model']}
            Platform: {row['platform_name']}
            Genre: {row['genre_name']}
            Current Price: €{row['products_price']}
            Used Price: €{row['products_used_price']}
            Margin: {row['products_margin']}%
            Stock: {row['products_quantity']}
            Barcode: {row['products_gtin']}
            """
            
            metadata = {
                "doc_type": "product",
                "model": row['products_model'],
                "platform": row['platform_name'],
                "genre": row['genre_name'],
                "current_price": float(row['products_price'] or 0),
                "used_price": float(row['products_used_price'] or 0)
            }
            
            doc = Document(
                text=content.strip(),
                metadata=metadata,
                doc_id=row['doc_id']
            )
            documents.append(doc)
        
        cursor.close()
        connection.close()
        
        return documents
    
    def build_index(self):
        """Build the complete RAG index"""
        print("Ingesting transaction history...")
        transaction_docs = self.ingest_transaction_history()
        print(f"Loaded {len(transaction_docs)} transaction documents")
        
        print("Ingesting product catalog...")
        product_docs = self.ingest_product_catalog()
        print(f"Loaded {len(product_docs)} product documents")
        
        all_documents = transaction_docs + product_docs
        
        print("Building vector index...")
        self.index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            response_mode="compact"
        )
        
        print("RAG system ready!")
    
    def query_pricing(self, game_title: str, platform: str, condition: str):
        """Query the RAG system for pricing insights"""
        if not self.query_engine:
            raise ValueError("RAG system not initialized. Call build_index() first.")
        
        query = f"""
        Analyze pricing for game "{game_title}" on {platform} platform in {condition} condition.
        Consider:
        1. Historical sales data and profit margins
        2. Current catalog pricing
        3. Market trends and demand
        4. Condition-based price adjustments
        
        Provide specific price recommendations with reasoning.
        """
        
        response = self.query_engine.query(query)
        return response

# Usage example
if __name__ == "__main__":
    mysql_config = {
        'user': 'your_username',
        'password': 'your_password',
        'host': 'localhost',
        'database': 'your_database'
    }
    
    rag_system = GamePricingRAG(mysql_config)
    rag_system.build_index()
    
    # Test query
    result = rag_system.query_pricing("Mario Kart 8 Deluxe", "Switch", "good")
    print(result)
```

## Deliverables & Success Criteria

### Week 1-2 Deliverables:
- [ ] Docker development environment with GPU support
- [ ] Enhanced MySQL schema with new tables
- [ ] Database migration scripts
- [ ] Environment verification scripts

### Week 3-4 Deliverables:
- [ ] Rasa framework configured and running
- [ ] Basic conversational flow implemented
- [ ] Training data created for game pricing intents
- [ ] Custom actions framework for database queries

### Week 5-6 Deliverables:
- [ ] LlamaIndex RAG system operational
- [ ] Data ingestion pipeline for existing database
- [ ] Vector store (ChromaDB) configured
- [ ] Query engine for intelligent price insights

### Success Criteria:
1. **Environment**: GPU environment properly configured and verified
2. **Database**: All new tables created and integrated with existing schema
3. **Rasa**: Basic chatbot responds to text queries about game prices
4. **RAG**: System can retrieve relevant pricing information from existing data
5. **Integration**: All components communicate successfully

### Testing Checklist:
- [ ] GPU acceleration working for ML models
- [ ] Database queries return expected results
- [ ] Rasa NLU correctly identifies pricing intents
- [ ] RAG system retrieves relevant documents
- [ ] End-to-end text query flow functional

## Next Phase Preview
Phase 2 will focus on computer vision integration, implementing Meta SAM for image segmentation, multi-OCR pipeline for text extraction, and vision-language models for game identification from photos.

## Team Roles & Responsibilities
- **Backend Developer**: Database schema and API development
- **ML Engineer**: RAG system and model integration
- **DevOps Engineer**: Environment setup and deployment
- **QA Engineer**: Testing framework and validation

## Risk Mitigation
- **GPU Resources**: Ensure cloud backup for GPU availability
- **Database Performance**: Monitor query performance with new schema
- **Model Dependencies**: Version lock all ML model dependencies
- **Data Quality**: Validate data migration and integrity
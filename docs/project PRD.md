

## Executive Summary

This PRD outlines the development of an intelligent chatbot system for second-hand game retailers that provides accurate price estimations by combining historical transaction data with real-time marketplace analysis and advanced computer vision capabilities.

**Key Value Proposition:**
- **Instant Price Estimation**: Get accurate pricing through text queries or photo uploads
- **Multi-Source Intelligence**: Combines your MySQL transaction history with live marketplace data from eBay, Rakuten France, and Vinted
- **Advanced Computer Vision**: Photo-based game identification using Meta's SAM and modern OCR models
- **Open-Source Foundation**: Built entirely with open-source technologies for maximum flexibility and cost-effectiveness

---

## Architecture Overview

### Core Technology Stack

**Conversational AI Framework:**
- **Primary**: Rasa Open Source - Advanced NLU and dialogue management
- **Alternative**: Botpress for visual development interface

**Retrieval-Augmented Generation (RAG):**
- **LlamaIndex**: Core RAG framework for intelligent data retrieval and synthesis
- **Vector Database**: ChromaDB or PostgreSQL with pgvector for embeddings storage
- **Embeddings**: HuggingFace Sentence Transformers (open-source)

**Computer Vision Pipeline:**
- **Meta's SAM (Segment Anything Model)**: Advanced image segmentation for game box identification
- **OCR Engine**: Surya, EasyOCR, or TrOCR for text extraction from game boxes
- **Vision Language Model**: Qwen2.5-VL or Moondream2 for multimodal understanding

**Data Integration:**
- **Database**: MySQL (existing) + Vector store for embeddings
- **Marketplace APIs**: Custom scrapers for eBay, Rakuten France, Vinted
- **Game Database**: TheGamesDB API for comprehensive game information

---

## System Components

### 1. Conversational Interface Layer

**Rasa NLU Pipeline:**
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
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
```

**Intent Recognition:**
- `price_inquiry_text`: "What's the price of Mario Kart 8 Deluxe for Switch?"
- `price_inquiry_photo`: "Can you price this game?" [with photo upload]
- `condition_inquiry`: "What's this worth in mint condition?"
- `platform_specific`: "How much for PS5 games?"
- `market_comparison`: "Compare prices across platforms"
- `historical_trend`: "Show me price history for this game"

**Entity Extraction:**
- Game titles, platforms, conditions, release years
- Price ranges, currencies, marketplace preferences

### 2. Computer Vision Pipeline

**Photo Processing Workflow:**

1. **Image Preprocessing:**
   ```python
   # OpenCV-based preprocessing
   - Image normalization and enhancement
   - Perspective correction for angled photos
   - Noise reduction and sharpening
   ```

2. **SAM-Powered Segmentation:**
   ```python
   # Meta's Segment Anything Model
   from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
   
   sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
   mask_generator = SamAutomaticMaskGenerator(sam)
   
   # Automatic segmentation to isolate game box
   masks = mask_generator.generate(image)
   ```

3. **OCR Text Extraction:**
   ```python
   # Multi-OCR approach for maximum accuracy
   - Primary: Surya (90+ languages, high accuracy)
   - Fallback: EasyOCR for problematic cases
   - Barcode: Specialized UPC/EAN detection
   ```

4. **Vision-Language Understanding:**
   ```python
   # Qwen2.5-VL for multimodal reasoning
   - Game title extraction
   - Platform identification
   - Condition assessment
   - Edition/variant detection
   ```

### 3. LlamaIndex RAG System

**Data Ingestion Pipeline:**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Transaction History Ingestion
transaction_reader = DatabaseReader(
    sql_database=mysql_connection,
    query="SELECT * FROM transactions WHERE date >= '2023-01-01'"
)

# Marketplace Data Ingestion
marketplace_reader = APIReader(
    apis=["ebay_completed", "rakuten_fr", "vinted_sold"]
)

# Create unified index
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
index = VectorStoreIndex.from_documents(
    documents=[transaction_reader, marketplace_reader],
    embed_model=embed_model
)
```

**Query Processing:**

```python
# Intelligent retrieval with metadata filtering
query_engine = index.as_query_engine(
    similarity_top_k=10,
    filters=MetadataFilters([
        MetadataFilter(key="platform", value=detected_platform),
        MetadataFilter(key="condition", value=estimated_condition),
        MetadataFilter(key="date_range", value="last_6_months")
    ])
)

# Multi-step reasoning
response = query_engine.query(
    f"What is the current market price for {game_title} on {platform} "
    f"in {condition} condition, considering recent sales data?"
)
```

### 4. Marketplace Data Integration

**eBay Integration:**
```python
# Official eBay API + Scraping hybrid
class eBayPriceAnalyzer:
    def __init__(self):
        self.api_client = eBayAPIClient()
        self.scraper = eBayScrapingClient()
    
    def get_completed_listings(self, game_title, platform):
        # Official API for basic search
        api_results = self.api_client.search_completed(game_title)
        
        # Scraping for detailed condition data
        detailed_results = self.scraper.get_condition_prices(api_results)
        
        return self.analyze_price_distribution(detailed_results)
```

**Rakuten France Integration:**
```python
# Using Piloterr API for Rakuten data
class RakutenFranceAnalyzer:
    def __init__(self):
        self.api_client = PiloterrClient(api_key=PILOTERR_KEY)
    
    def search_games(self, query):
        return self.api_client.rakuten_search(
            query=query,
            country="france",
            category="video_games"
        )
```

**Vinted Integration:**
```python
# Open-source vinted-scraper package
from vinted_scraper import VintedScraper

class VintedGameAnalyzer:
    def __init__(self):
        self.scraper = VintedScraper("https://www.vinted.fr")
    
    def search_games(self, game_title):
        params = {
            "search_text": f"{game_title} jeu video",
            "catalog[]": "76"  # Video games category
        }
        return self.scraper.search(params)
```

### 5. Intelligent Pricing Algorithm

**Multi-Factor Price Calculation Using Your Data:**

```python
class IntelligentPricingEngine:
    def __init__(self, llamaindex_engine, mysql_connection):
        self.rag_engine = llamaindex_engine
        self.db = mysql_connection
        
    def calculate_price_estimate(self, products_model, condition, market_data):
        # 1. Historical analysis from your sales data
        historical_data = self.get_historical_sales_data(products_model, condition)
        
        # 2. Current market analysis via RAG
        market_context = self.rag_engine.query(
            f"Analyze current market pricing for product {products_model} "
            f"in {condition} condition considering recent marketplace data"
        )
        
        # 3. Your specific business logic
        profit_margins = self.calculate_profit_margins(products_model)
        current_stock_price = self.get_current_stock_price(products_model)
        
        # 4. Weighted calculation
        weights = {
            'your_historical_sales': 0.35,    # Your actual transaction history
            'your_current_price': 0.25,       # Your current products_used_price
            'marketplace_sold': 0.25,         # eBay/Rakuten sold listings
            'marketplace_current': 0.15       # Current marketplace listings
        }
        
        final_estimate = self.weighted_price_calculation(
            historical_data, current_stock_price, market_context, weights
        )
        
        return {
            'estimated_price': final_estimate,
            'your_current_price': current_stock_price,
            'your_avg_sold_price': historical_data['avg_price'],
            'your_profit_margin': profit_margins,
            'market_comparison': self.compare_to_market(final_estimate, market_data),
            'confidence_score': self.calculate_confidence(historical_data, market_data),
            'recommendation': self.generate_pricing_recommendation(final_estimate, current_stock_price)
        }
    
    def get_historical_sales_data(self, products_model, condition):
        """Extract your actual sales history for this product"""
        query = """
        SELECT 
            AVG(s.items_sold_price_stock) as avg_sale_price,
            AVG(s.items_sold_price_gross) as avg_cost,
            COUNT(*) as sales_count,
            MAX(STR_TO_DATE(s.items_sold_date_in, '%Y%m%d')) as last_sale_date,
            STDDEV(s.items_sold_price_stock) as price_variance
        FROM netshop_items_sold s
        WHERE s.items_sold_model = %s 
        AND s.items_sold_condition_grade = %s
        AND STR_TO_DATE(s.items_sold_date_in, '%Y%m%d') >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        """
        
        with self.db.cursor() as cursor:
            cursor.execute(query, (products_model, condition))
            result = cursor.fetchone()
            
        return {
            'avg_price': result['avg_sale_price'] or 0,
            'avg_cost': result['avg_cost'] or 0,
            'sales_count': result['sales_count'],
            'last_sale_date': result['last_sale_date'],
            'price_variance': result['price_variance'] or 0,
            'profit_margin': ((result['avg_sale_price'] or 0) - (result['avg_cost'] or 0)) / (result['avg_sale_price'] or 1) * 100
        }
    
    def get_current_stock_price(self, products_model):
        """Get your current pricing from products table"""
        query = """
        SELECT 
            p.products_used_price,
            p.products_used_updated,
            p.products_price,
            p.products_margin,
            p.products_quantity,
            pd.products_name,
            plat.platform_name
        FROM products p
        LEFT JOIN products_description pd ON p.products_id = pd.products_id
        LEFT JOIN platforms plat ON p.products_platform = plat.platform_id
        WHERE p.products_model = %s
        """
        
        with self.db.cursor() as cursor:
            cursor.execute(query, (products_model,))
            return cursor.fetchone()
```

**LlamaIndex Integration with Your Schema:**

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.readers.database import DatabaseReader

class GamePricingRAG:
    def __init__(self, mysql_connection):
        self.db_connection = mysql_connection
        self.setup_rag_system()
    
    def setup_rag_system(self):
        """Initialize LlamaIndex with your existing data"""
        
        # Create documents from your sales history
        sales_reader = DatabaseReader(
            sql_database=self.db_connection,
            query="""
            SELECT 
                CONCAT('SALE_', s.items_sold_id) as doc_id,
                CONCAT(
                    'Product: ', pd.products_name, 
                    ', Model: ', s.items_sold_model,
                    ', Platform: ', plat.platform_name,
                    ', Condition: ', s.items_sold_condition_grade,
                    ', Sale Price: €', s.items_sold_price_stock,
                    ', Cost: €', s.items_sold_price_gross,
                    ', Date: ', STR_TO_DATE(s.items_sold_date_in, '%Y%m%d'),
                    ', Profit: €', (s.items_sold_price_stock - s.items_sold_price_gross)
                ) as content
            FROM netshop_items_sold s
            JOIN products p ON s.items_sold_model = p.products_model
            JOIN products_description pd ON p.products_id = pd.products_id
            LEFT JOIN platforms plat ON p.products_platform = plat.platform_id
            WHERE s.items_sold_condition_grade IS NOT NULL
            AND STR_TO_DATE(s.items_sold_date_in, '%Y%m%d') >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
            """
        )
        
        # Create documents from current product catalog
        catalog_reader = DatabaseReader(
            sql_database=self.db_connection,
            query="""
            SELECT 
                CONCAT('PRODUCT_', p.products_model) as doc_id,
                CONCAT(
                    'Product: ', pd.products_name,
                    ', Model: ', p.products_model,
                    ', Platform: ', plat.platform_name,
                    ', Genre: ', g.genre_name,
                    ', Current Price: €', p.products_price,
                    ', Used Price: €', p.products_used_price,
                    ', Margin: ', p.products_margin,
                    ', Stock: ', p.products_quantity,
                    ', Barcode: ', p.products_gtin,
                    ', Sales Velocity: ', p.products_sold_velocity
                ) as content
            FROM products p
            JOIN products_description pd ON p.products_id = pd.products_id
            LEFT JOIN platforms plat ON p.products_platform = plat.platform_id
            LEFT JOIN genres g ON p.products_genre = g.genre_id
            WHERE p.products_status = 1
            """
        )
        
        # Load marketplace data
        marketplace_reader = DatabaseReader(
            sql_database=self.db_connection,
            query="""
            SELECT 
                CONCAT('MARKET_', id) as doc_id,
                CONCAT(
                    'Marketplace: ', source,
                    ', Product: ', products_model,
                    ', Price: €', price,
                    ', Condition: ', condition_grade,
                    ', Date: ', listing_date,
                    ', Status: ', IF(is_sold, 'SOLD', 'LISTED')
                ) as content
            FROM marketplace_prices
            WHERE listing_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
            """
        )
        
        # Combine all documents and create index
        all_documents = []
        all_documents.extend(sales_reader.load_data())
        all_documents.extend(catalog_reader.load_data())
        all_documents.extend(marketplace_reader.load_data())
        
        # Create vector index
        self.index = VectorStoreIndex.from_documents(all_documents)
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            response_mode="compact"
        )
    
    def get_pricing_insights(self, products_model, condition):
        """Use RAG to get comprehensive pricing insights"""
        query = f"""
        Analyze the pricing for product model {products_model} in {condition} condition.
        Consider:
        1. Historical sales performance and profit margins
        2. Current market prices from different sources
        3. Pricing trends and velocity
        4. Competitive positioning
        
        Provide specific price recommendations with reasoning.
        """
        
        response = self.query_engine.query(query)
        return response
```

---

## Database Schema Integration

### Working with Your Existing Schema

**Existing Tables Analysis:**
- `netshop_items_sold`: Your transaction history (sales data)
- `products`: Your product catalog with detailed information

**Key Data Mappings:**
```sql
-- Your existing structure provides:
SELECT 
    p.products_model,
    pd.products_name,
    p.products_platform,  -- Platform info
    p.products_genre,     -- Genre classification
    p.products_gtin,      -- Barcode/UPC for identification
    s.items_sold_condition_grade,  -- Condition data
    s.items_sold_price_stock,      -- Sale price to customer
    s.items_sold_price_gross,      -- Your cost (profit calculation)
    s.items_sold_date_in,          -- Sale date
    p.products_used_price,         -- Current used price
    p.products_used_updated        -- Last price update
FROM netshop_items_sold s
JOIN products p ON s.items_sold_model = p.products_model
JOIN products_description pd ON p.products_id = pd.products_id
WHERE s.items_sold_condition_grade IS NOT NULL
```

### Minimal New Tables Needed

```sql
-- Lookup tables for better data normalization
CREATE TABLE platforms (
    platform_id INT PRIMARY KEY,
    platform_name VARCHAR(50) NOT NULL,
    platform_short VARCHAR(10) NOT NULL,
    UNIQUE KEY (platform_name)
);

CREATE TABLE genres (
    genre_id INT PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL,
    UNIQUE KEY (genre_name)
);

-- Enhanced marketplace data storage
CREATE TABLE marketplace_prices (
    id INT PRIMARY KEY AUTO_INCREMENT,
    products_model VARCHAR(15) NOT NULL,  -- Link to your products table
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

-- Photo analysis cache for computer vision results
CREATE TABLE photo_analysis_cache (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_hash VARCHAR(64) UNIQUE,
    detected_model VARCHAR(15),  -- Links to products_model
    detected_platform_id INT,
    detected_condition VARCHAR(50),
    detected_gtin VARCHAR(13),   -- Barcode if detected
    confidence_score DECIMAL(3,2),
    analysis_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detected_platform_id) REFERENCES platforms(platform_id),
    INDEX idx_hash (image_hash),
    INDEX idx_model (detected_model)
);

-- LlamaIndex document embeddings (can be separate database if needed)
CREATE TABLE ai_document_embeddings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    document_id VARCHAR(100) UNIQUE,
    document_type ENUM('transaction', 'product', 'marketplace_listing'),
    source_id VARCHAR(50),  -- products_model or items_sold_id
    content TEXT,
    embedding_vector JSON,  -- Store as JSON array since MySQL doesn't have vector type
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

---

## API Design

### RESTful API Endpoints

```python
# FastAPI application structure
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="Game Pricing Chatbot API")

class PriceQuery(BaseModel):
    query: str
    context: Optional[dict] = None

class PriceResponse(BaseModel):
    estimated_price: float
    price_range: dict
    confidence_score: float
    sources: list
    recommendations: list

@app.post("/chat/message")
async def process_message(query: PriceQuery):
    """Process text-based pricing queries"""
    pass

@app.post("/chat/photo")
async def process_photo(file: UploadFile = File(...)):
    """Process photo-based pricing queries"""
    pass

@app.get("/games/search")
async def search_games(title: str, platform: Optional[str] = None):
    """Search game catalog"""
    pass

@app.get("/prices/history/{game_id}")
async def get_price_history(game_id: int):
    """Get historical pricing data"""
    pass
```

### Chatbot Conversation Flow

```yaml
# Rasa domain.yml
version: "3.1"

intents:
  - price_inquiry_text
  - price_inquiry_photo
  - condition_inquiry
  - greet
  - goodbye

entities:
  - game_title
  - platform
  - condition

responses:
  utter_greet:
  - text: "Hello! I can help you price second-hand games. You can ask me about specific games or upload a photo of the game box."
  
  utter_price_result:
  - text: "Based on current market data, {game_title} for {platform} in {condition} condition is estimated at €{price} (±{range}). Confidence: {confidence}%"
  
  utter_ask_photo_upload:
  - text: "Please upload a clear photo of the game box (front cover preferred) and I'll identify it and provide pricing."

actions:
  - action_process_price_query
  - action_analyze_photo
  - action_get_market_data
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (4-6 weeks)
- Set up Rasa framework with basic NLU
- Implement LlamaIndex RAG system
- Create MySQL schema enhancements
- Build basic marketplace API integrations
- Develop core pricing algorithm

### Phase 2: Computer Vision Integration (3-4 weeks)
- Integrate Meta's SAM for image segmentation
- Implement multi-OCR pipeline
- Add vision-language model for game identification
- Create photo upload interface
- Build image analysis caching system

### Phase 3: Advanced Features (2-3 weeks)
- Implement real-time marketplace monitoring
- Add price trend analysis
- Create recommendation engine
- Build admin dashboard for data management
- Optimize performance and scaling

### Phase 4: Testing & Deployment (2-3 weeks)
- Comprehensive testing with real game images
- Performance optimization
- Security audit
- Documentation and training materials
- Production deployment

---

## Technical Considerations

### Performance Optimization

**Caching Strategy:**
- Redis for frequent queries
- Image analysis results cached by hash
- Marketplace data refreshed every 6 hours
- Price calculations cached for 24 hours

**Scalability:**
- Horizontal scaling with Docker containers
- Load balancing for multiple chat instances
- Async processing for marketplace data updates
- CDN for image storage and processing

### Security & Privacy

**Data Protection:**
- No PII storage from uploaded images
- Encrypted API keys and database connections
- Rate limiting on photo uploads
- GDPR-compliant data retention policies

**Input Validation:**
- File type restrictions (JPEG, PNG only)
- Image size limits (max 10MB)
- SQL injection prevention
- XSS protection on chat interface

### Error Handling

**Graceful Degradation:**
- Fallback OCR engines if primary fails
- Alternative pricing sources if APIs are down
- Confidence scoring for uncertain identifications
- Clear error messages for users

---

## Cost Analysis

### Open-Source Components (Free)
- Rasa framework
- LlamaIndex
- Meta's SAM
- OCR engines (Surya, EasyOCR)
- Qwen2.5-VL vision model
- ChromaDB vector database

### Operational Costs (Monthly)
- **Hosting**: €50-100 (VPS with GPU support)
- **API Calls**: €30-50 (marketplace scraping)
- **Storage**: €10-20 (images and embeddings)
- **Total Estimated**: €90-170/month

### Cost Savings vs. Commercial Alternatives
- OpenAI GPT-4V: €500-1000/month
- Google Vision API: €200-400/month
- Commercial OCR: €300-600/month
- **Savings**: 70-80% reduction in AI costs

---

## Success Metrics

### Accuracy Metrics
- **Photo Identification**: >90% correct game detection
- **Price Estimation**: ±15% of actual market price
- **Condition Assessment**: 85% accuracy vs. manual grading

### Performance Metrics
- **Response Time**: <3 seconds for text queries
- **Photo Processing**: <10 seconds end-to-end
- **System Uptime**: >99.5%

### Business Metrics
- **User Adoption**: Track daily active users
- **Query Volume**: Monitor pricing requests
- **Accuracy Feedback**: User satisfaction ratings

---

## Future Enhancements

### Advanced Features
- **Multi-language Support**: French, German, Spanish
- **Voice Interface**: Audio queries via speech recognition
- **AR Integration**: Real-time price overlay using phone camera
- **Batch Processing**: Upload multiple game photos at once

### AI Improvements
- **Custom Game Detection Model**: Train on your specific inventory
- **Predictive Pricing**: ML models for price forecasting
- **Automated Buying Recommendations**: Identify underpriced items
- **Market Sentiment Analysis**: Social media and forum monitoring

### Integration Possibilities
- **POS System Integration**: Direct pricing in your sales system
- **Inventory Management**: Track stock levels and values
- **Financial Reporting**: Automated profit/loss calculations
- **Customer App**: Let customers check trade-in values

---

## Conclusion

This comprehensive solution leverages cutting-edge open-source AI technologies to create a powerful, cost-effective game pricing system. By combining LlamaIndex's RAG capabilities with Meta's SAM computer vision and modern NLU frameworks, you'll have a competitive advantage in the second-hand gaming market.

The system's modular architecture allows for incremental development and easy scaling, while the open-source foundation ensures long-term flexibility and cost control. With proper implementation, this chatbot will provide accurate, instant pricing estimates that improve your operational efficiency and customer experience.

**Next Steps:**
1. Review and approve this PRD
2. Set up development environment
3. Begin Phase 1 implementation
4. Gather initial game photos for testing
5. Define success criteria and testing protocols
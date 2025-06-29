-- PriceVision Database Schema Enhancement
-- Phase 1: Core Infrastructure Setup

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
    INDEX idx_session (session_id),
    INDEX idx_model (products_model)
);

-- Create a basic products table if it doesn't exist (for foreign key references)
CREATE TABLE IF NOT EXISTS products (
    products_model VARCHAR(15) PRIMARY KEY,
    products_name VARCHAR(255),
    platform_id INT,
    genre_id INT,
    release_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (platform_id) REFERENCES platforms(platform_id),
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
);

-- Add foreign key constraints after products table exists
ALTER TABLE marketplace_prices 
ADD CONSTRAINT fk_marketplace_products 
FOREIGN KEY (products_model) REFERENCES products(products_model);

ALTER TABLE chatbot_conversations 
ADD CONSTRAINT fk_conversations_products 
FOREIGN KEY (products_model) REFERENCES products(products_model);
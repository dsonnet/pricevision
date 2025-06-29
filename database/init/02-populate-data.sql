-- PriceVision Initial Data Population
-- Phase 1: Core Infrastructure Setup

-- Populate platforms table with common gaming platforms
INSERT INTO platforms (platform_name, platform_short) VALUES
('PlayStation 5', 'PS5'),
('PlayStation 4', 'PS4'),
('PlayStation 3', 'PS3'),
('PlayStation 2', 'PS2'),
('PlayStation 1', 'PS1'),
('Xbox Series X/S', 'XSX'),
('Xbox One', 'XB1'),
('Xbox 360', 'X360'),
('Xbox', 'XB'),
('Nintendo Switch', 'NSW'),
('Nintendo Wii U', 'WIIU'),
('Nintendo Wii', 'WII'),
('Nintendo GameCube', 'GCN'),
('Nintendo 64', 'N64'),
('Super Nintendo', 'SNES'),
('Nintendo Entertainment System', 'NES'),
('Nintendo 3DS', '3DS'),
('Nintendo DS', 'NDS'),
('Game Boy Advance', 'GBA'),
('Game Boy Color', 'GBC'),
('Game Boy', 'GB'),
('PC', 'PC'),
('Steam Deck', 'DECK'),
('PlayStation Portable', 'PSP'),
('PlayStation Vita', 'VITA');

-- Populate genres table with common game genres
INSERT INTO genres (genre_name) VALUES
('Action'),
('Adventure'),
('Role-Playing (RPG)'),
('Strategy'),
('Simulation'),
('Sports'),
('Racing'),
('Fighting'),
('Shooter'),
('Platform'),
('Puzzle'),
('Horror'),
('Survival'),
('Sandbox'),
('MMORPG'),
('Battle Royale'),
('Roguelike'),
('Visual Novel'),
('Music/Rhythm'),
('Educational'),
('Party'),
('Arcade'),
('Stealth'),
('Tower Defense'),
('Card Game');

-- Insert sample products for testing
INSERT INTO products (products_model, products_name, platform_id, genre_id, release_date) VALUES
('TLOU2-PS4', 'The Last of Us Part II', 2, 1, '2020-06-19'),
('GOW-PS4', 'God of War (2018)', 2, 1, '2018-04-20'),
('ZELDA-NSW', 'The Legend of Zelda: Breath of the Wild', 10, 2, '2017-03-03'),
('HALO-XB1', 'Halo 5: Guardians', 7, 9, '2015-10-27'),
('MARIO-NSW', 'Super Mario Odyssey', 10, 10, '2017-10-27'),
('CYBERPUNK-PC', 'Cyberpunk 2077', 22, 3, '2020-12-10'),
('FIFA23-PS5', 'FIFA 23', 1, 6, '2022-09-30'),
('COD-MW2-PS5', 'Call of Duty: Modern Warfare II', 1, 9, '2022-10-28'),
('ELDEN-PS5', 'Elden Ring', 1, 3, '2022-02-25'),
('POKEMON-NSW', 'Pokémon Scarlet and Violet', 10, 3, '2022-11-18');

-- Insert sample marketplace prices for testing
INSERT INTO marketplace_prices (products_model, source, listing_title, price, condition_grade, listing_date, is_sold, currency) VALUES
('TLOU2-PS4', 'ebay', 'The Last of Us Part II PS4 - Excellent Condition', 25.99, 'near_mint', '2024-01-15', TRUE, 'EUR'),
('TLOU2-PS4', 'rakuten_fr', 'TLOU2 PlayStation 4', 22.50, 'good', '2024-01-20', TRUE, 'EUR'),
('GOW-PS4', 'ebay', 'God of War PS4 Game', 19.99, 'good', '2024-01-18', TRUE, 'EUR'),
('ZELDA-NSW', 'ebay', 'Zelda Breath of the Wild Nintendo Switch', 45.00, 'mint', '2024-01-22', FALSE, 'EUR'),
('HALO-XB1', 'ebay', 'Halo 5 Guardians Xbox One', 15.99, 'good', '2024-01-25', TRUE, 'EUR'),
('MARIO-NSW', 'rakuten_fr', 'Super Mario Odyssey Switch', 42.99, 'near_mint', '2024-01-28', FALSE, 'EUR'),
('CYBERPUNK-PC', 'ebay', 'Cyberpunk 2077 PC Steam Key', 29.99, 'mint', '2024-02-01', TRUE, 'EUR'),
('FIFA23-PS5', 'ebay', 'FIFA 23 PlayStation 5', 35.00, 'near_mint', '2024-02-05', FALSE, 'EUR'),
('ELDEN-PS5', 'rakuten_fr', 'Elden Ring PS5 Edition', 49.99, 'mint', '2024-02-08', FALSE, 'EUR'),
('POKEMON-NSW', 'ebay', 'Pokemon Scarlet Nintendo Switch', 38.50, 'good', '2024-02-10', TRUE, 'EUR');

-- Insert sample document embeddings for RAG testing
INSERT INTO ai_document_embeddings (document_id, document_type, source_id, content, metadata) VALUES
('prod_tlou2_desc', 'product', 'TLOU2-PS4', 'The Last of Us Part II is an action-adventure survival horror game. Set four years after the events of the first game, players control 19-year-old Ellie, who comes into conflict with a mysterious cult in a post-apocalyptic United States.', '{"category": "product_description", "platform": "PS4", "genre": "Action"}'),
('prod_gow_desc', 'product', 'GOW-PS4', 'God of War is an action-adventure game featuring Norse mythology. Players control Kratos and his son Atreus as they journey through the nine realms. The game features over-the-shoulder third-person combat.', '{"category": "product_description", "platform": "PS4", "genre": "Action"}'),
('market_trend_action', 'marketplace_listing', 'trend_2024_q1', 'Action games on PlayStation 4 show strong resale value, with titles like God of War and The Last of Us Part II maintaining 60-70% of their original retail price in good condition.', '{"category": "market_analysis", "period": "2024_Q1", "platform": "PS4"}');

-- Insert sample conversation history for testing
INSERT INTO chatbot_conversations (session_id, user_query, query_type, detected_intent, extracted_entities, products_model, estimated_price, confidence_score, response_data, processing_time_ms) VALUES
('sess_001', 'What is the price of The Last of Us Part II for PS4?', 'text', 'price_inquiry', '{"game": "The Last of Us Part II", "platform": "PS4"}', 'TLOU2-PS4', 24.25, 0.92, '{"sources": ["ebay", "rakuten_fr"], "price_range": "22.50-25.99"}', 1250),
('sess_002', 'How much is God of War worth on PlayStation 4?', 'text', 'price_inquiry', '{"game": "God of War", "platform": "PlayStation 4"}', 'GOW-PS4', 19.99, 0.88, '{"sources": ["ebay"], "condition": "good"}', 980),
('sess_003', 'Price check for Zelda Breath of the Wild Switch', 'text', 'price_inquiry', '{"game": "Zelda Breath of the Wild", "platform": "Switch"}', 'ZELDA-NSW', 45.00, 0.95, '{"sources": ["ebay"], "condition": "mint", "availability": "in_stock"}', 1100);

-- Create indexes for better performance
CREATE INDEX idx_products_platform ON products(platform_id);
CREATE INDEX idx_products_genre ON products(genre_id);
CREATE INDEX idx_marketplace_created ON marketplace_prices(created_at);
CREATE INDEX idx_conversations_created ON chatbot_conversations(created_at);
CREATE INDEX idx_embeddings_created ON ai_document_embeddings(created_at);
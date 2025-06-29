"""
PriceVision FastAPI Application
RTX 3050 optimized main application with chatbot and photo analysis
"""

import logging
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import our modules
from .config import config
from .database import initialize_database, get_mysql_session, get_redis_client
from .memory_manager import memory_manager, initialize_memory_manager
from .rag_system import initialize_rag_system, query_rag_system, rag_context
from .photo_analysis import initialize_photo_analyzer, analyze_game_photo, photo_analysis_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot response")
    session_id: str = Field(..., description="Chat session ID")
    confidence: float = Field(..., description="Response confidence")
    suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    timestamp: str = Field(..., description="Response timestamp")

class PhotoAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Analysis success status")
    game_info: Dict[str, Any] = Field(..., description="Extracted game information")
    confidence_score: float = Field(..., description="Analysis confidence")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    analysis_time: float = Field(..., description="Processing time in seconds")

class PriceInquiry(BaseModel):
    game_title: str = Field(..., description="Game title")
    platform: str = Field(..., description="Gaming platform")
    condition: str = Field(..., description="Game condition")
    marketplace: Optional[str] = Field(None, description="Target marketplace")

class PriceResponse(BaseModel):
    estimated_price: float = Field(..., description="Estimated price")
    price_range: Dict[str, float] = Field(..., description="Price range (low, high)")
    marketplace_comparison: Dict[str, Dict[str, float]] = Field(..., description="Marketplace prices")
    confidence: float = Field(..., description="Price confidence")
    factors: List[str] = Field(..., description="Price factors")

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    try:
        logger.info("Starting PriceVision application...")
        
        # Initialize RTX 3050 optimized components
        await initialize_memory_manager()
        await initialize_database()
        
        # Initialize AI components with memory management
        logger.info("Initializing AI components...")
        await initialize_rag_system()
        await initialize_photo_analyzer()
        
        logger.info("PriceVision application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down PriceVision application...")
        await memory_manager.cleanup_all()

# Create FastAPI app with RTX 3050 optimizations
app = FastAPI(
    title="PriceVision API",
    description="Intelligent chatbot for second-hand game pricing with photo analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found, skipping static file mounting")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        async with get_mysql_session() as session:
            pass
        
        # Check Redis connectivity
        redis_client = await get_redis_client()
        await redis_client.ping()
        
        # Check GPU status
        gpu_status = await memory_manager.get_gpu_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "redis": "connected",
            "gpu": gpu_status,
            "memory_usage": await memory_manager.get_memory_usage()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(message: ChatMessage):
    """Chat with the PriceVision bot"""
    try:
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = message.session_id or f"session_{int(time.time())}"
        
        # Process message with Rasa (mock implementation for now)
        bot_response = await process_chat_message(
            message.message,
            session_id,
            message.context
        )
        
        # Generate suggestions
        suggestions = generate_chat_suggestions(message.message, bot_response)
        
        response = ChatResponse(
            response=bot_response,
            session_id=session_id,
            confidence=0.8,  # This would come from actual Rasa confidence
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
        
        # Log conversation
        await log_conversation(session_id, message.message, bot_response)
        
        logger.info(f"Chat processed in {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/chat/photo", response_model=PhotoAnalysisResponse)
async def analyze_photo_for_chat(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze uploaded photo for chat context"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Analyze photo with memory management
        async with photo_analysis_context():
            analysis_result = await analyze_game_photo(image_data, file.filename)
        
        if not analysis_result.get("success"):
            raise HTTPException(status_code=400, detail="Photo analysis failed")
        
        response = PhotoAnalysisResponse(
            success=analysis_result["success"],
            game_info=analysis_result["game_info"],
            confidence_score=analysis_result["confidence_score"],
            recommendations=analysis_result["recommendations"],
            analysis_time=analysis_result["analysis_time"]
        )
        
        # Store analysis result in session context
        if session_id:
            await store_photo_context(session_id, analysis_result)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Photo analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Photo analysis failed: {str(e)}")

# Price inquiry endpoints
@app.post("/price/inquiry", response_model=PriceResponse)
async def get_price_estimate(inquiry: PriceInquiry):
    """Get price estimate for a game"""
    try:
        # Query RAG system for pricing information
        async with rag_context():
            rag_query = f"What is the current market price for {inquiry.game_title} on {inquiry.platform} in {inquiry.condition} condition?"
            rag_response = await query_rag_system(rag_query, {
                "game_title": inquiry.game_title,
                "platform": inquiry.platform,
                "condition": inquiry.condition
            })
        
        # Calculate price estimate (mock implementation)
        price_data = await calculate_price_estimate(inquiry)
        
        response = PriceResponse(
            estimated_price=price_data["estimated_price"],
            price_range=price_data["price_range"],
            marketplace_comparison=price_data["marketplace_comparison"],
            confidence=price_data["confidence"],
            factors=price_data["factors"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Price inquiry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Price inquiry failed: {str(e)}")

@app.get("/price/trends/{game_title}")
async def get_price_trends(game_title: str, platform: Optional[str] = None):
    """Get price trends for a specific game"""
    try:
        # Query historical price data (mock implementation)
        trends_data = await get_game_price_trends(game_title, platform)
        
        return {
            "game_title": game_title,
            "platform": platform,
            "trends": trends_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Price trends query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Price trends query failed: {str(e)}")

# Market analysis endpoints
@app.get("/market/analysis")
async def get_market_analysis():
    """Get current market analysis"""
    try:
        # Query RAG system for market insights
        async with rag_context():
            market_query = "What are the current trends in the second-hand gaming market?"
            market_response = await query_rag_system(market_query)
        
        analysis = {
            "market_trends": {
                "hot_genres": ["Action", "RPG", "Adventure"],
                "trending_platforms": ["Nintendo Switch", "PlayStation 5"],
                "price_direction": "stable",
                "demand_level": "high"
            },
            "insights": market_response.get("answer", "Market analysis unavailable"),
            "confidence": market_response.get("confidence", 0.5),
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")

# System monitoring endpoints
@app.get("/system/status")
async def get_system_status():
    """Get system status and performance metrics"""
    try:
        gpu_status = await memory_manager.get_gpu_status()
        memory_usage = await memory_manager.get_memory_usage()
        
        return {
            "gpu": gpu_status,
            "memory": memory_usage,
            "active_models": memory_manager.loaded_models,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status query failed: {e}")
        raise HTTPException(status_code=500, detail=f"System status query failed: {str(e)}")

@app.post("/system/cleanup")
async def trigger_cleanup():
    """Trigger system cleanup"""
    try:
        await memory_manager.cleanup_all()
        return {
            "status": "cleanup_completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"System cleanup failed: {str(e)}")

# Helper functions
async def process_chat_message(message: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Process chat message with Rasa (mock implementation)"""
    try:
        # This would integrate with actual Rasa server
        # For now, provide intelligent mock responses
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['price', 'cost', 'worth', 'value']):
            return "I can help you determine the value of your game! Please tell me the game title, platform, and condition, or upload a photo for analysis."
        
        elif any(word in message_lower for word in ['photo', 'picture', 'image', 'upload']):
            return "Great! Please upload a photo of your game and I'll analyze it to identify the title, platform, and condition for accurate pricing."
        
        elif any(word in message_lower for word in ['sell', 'selling', 'marketplace']):
            return "I can recommend the best marketplace for selling your game! Different platforms work better for different types of games. What game are you looking to sell?"
        
        elif any(word in message_lower for word in ['condition', 'quality']):
            return "Game condition significantly affects price. I can assess condition from photos or help you determine it: Mint (like new), Excellent (minor wear), Good (normal wear), Fair (noticeable wear), or Poor (significant damage)."
        
        else:
            return "Hi! I'm PriceVision, your game pricing assistant. I can help you determine game values, analyze photos, compare marketplace prices, and recommend where to sell. How can I help you today?"
    
    except Exception as e:
        logger.error(f"Chat message processing failed: {e}")
        return "I'm having trouble processing your message right now. Please try again."

def generate_chat_suggestions(user_message: str, bot_response: str) -> List[str]:
    """Generate contextual chat suggestions"""
    suggestions = []
    
    message_lower = user_message.lower()
    
    if 'price' in message_lower:
        suggestions = [
            "Upload a photo for analysis",
            "Compare marketplace prices",
            "Get selling recommendations",
            "Check market trends"
        ]
    elif 'photo' in message_lower:
        suggestions = [
            "Get price estimate",
            "Compare with similar games",
            "Find best marketplace",
            "Get selling tips"
        ]
    else:
        suggestions = [
            "Ask about game pricing",
            "Upload a game photo",
            "Get market analysis",
            "Learn about selling tips"
        ]
    
    return suggestions[:3]  # Limit to 3 suggestions

async def log_conversation(session_id: str, user_message: str, bot_response: str):
    """Log conversation for analytics"""
    try:
        async with get_mysql_session() as session:
            # This would insert into chatbot_conversations table
            # Simplified for now
            pass
        
        # Also cache recent conversations in Redis
        redis_client = await get_redis_client()
        conversation_key = f"conversation:{session_id}"
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        }
        
        await redis_client.lpush(conversation_key, json.dumps(conversation_data))
        await redis_client.ltrim(conversation_key, 0, 99)  # Keep last 100 messages
        await redis_client.expire(conversation_key, 86400)  # 24 hour expiry
        
    except Exception as e:
        logger.error(f"Conversation logging failed: {e}")

async def store_photo_context(session_id: str, analysis_result: Dict[str, Any]):
    """Store photo analysis result in session context"""
    try:
        redis_client = await get_redis_client()
        context_key = f"photo_context:{session_id}"
        
        await redis_client.setex(
            context_key,
            3600,  # 1 hour expiry
            json.dumps(analysis_result, default=str)
        )
        
    except Exception as e:
        logger.error(f"Photo context storage failed: {e}")

async def calculate_price_estimate(inquiry: PriceInquiry) -> Dict[str, Any]:
    """Calculate price estimate (mock implementation)"""
    try:
        # This would query actual pricing database
        # Mock calculation for now
        
        base_prices = {
            "call of duty": 45.99,
            "fifa": 35.99,
            "mario": 55.99,
            "zelda": 59.99,
            "pokemon": 49.99
        }
        
        # Find base price
        base_price = 39.99  # Default
        for game, price in base_prices.items():
            if game in inquiry.game_title.lower():
                base_price = price
                break
        
        # Condition multipliers
        condition_multipliers = {
            "mint": 1.2,
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.4
        }
        
        multiplier = condition_multipliers.get(inquiry.condition.lower(), 0.8)
        estimated_price = base_price * multiplier
        
        # Price range
        price_range = {
            "low": estimated_price * 0.8,
            "high": estimated_price * 1.2
        }
        
        # Marketplace comparison
        marketplace_comparison = {
            "eBay": {"price": estimated_price * 1.1, "fees": estimated_price * 0.1},
            "Amazon": {"price": estimated_price * 1.15, "fees": estimated_price * 0.15},
            "Facebook": {"price": estimated_price * 0.95, "fees": 0.0},
            "Local": {"price": estimated_price * 0.9, "fees": 0.0}
        }
        
        return {
            "estimated_price": round(estimated_price, 2),
            "price_range": {k: round(v, 2) for k, v in price_range.items()},
            "marketplace_comparison": {
                k: {kk: round(vv, 2) for kk, vv in v.items()}
                for k, v in marketplace_comparison.items()
            },
            "confidence": 0.75,
            "factors": [
                f"Base price for {inquiry.game_title}",
                f"Condition: {inquiry.condition}",
                f"Platform: {inquiry.platform}",
                "Current market trends"
            ]
        }
        
    except Exception as e:
        logger.error(f"Price calculation failed: {e}")
        return {
            "estimated_price": 0.0,
            "price_range": {"low": 0.0, "high": 0.0},
            "marketplace_comparison": {},
            "confidence": 0.0,
            "factors": ["Error in calculation"]
        }

async def get_game_price_trends(game_title: str, platform: Optional[str] = None) -> Dict[str, Any]:
    """Get price trends for a game (mock implementation)"""
    try:
        # Mock trend data
        import random
        
        # Generate mock historical data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        base_price = random.uniform(30, 60)
        
        trends = []
        for i, month in enumerate(months):
            price = base_price + random.uniform(-5, 5) + (i * 0.5)  # Slight upward trend
            trends.append({
                "month": month,
                "average_price": round(price, 2),
                "volume": random.randint(50, 200)
            })
        
        return {
            "historical_data": trends,
            "trend_direction": "stable",
            "volatility": "low",
            "recommendation": "Good time to sell"
        }
        
    except Exception as e:
        logger.error(f"Price trends query failed: {e}")
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    # RTX 3050 optimized server configuration
    uvicorn.run(
        "src.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=1,  # Single worker for RTX 3050 memory constraints
        reload=config.api.debug,
        log_level="info",
        access_log=True
    )
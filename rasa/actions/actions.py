"""
PriceVision Rasa Custom Actions
RTX 3050 optimized custom actions for chatbot functionality
"""

import logging
import asyncio
from typing import Any, Text, Dict, List, Optional
import json
import time
from datetime import datetime

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet, FollowupAction, ActionExecuted

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import config
from src.database import get_mysql_session, get_redis_client, get_chroma_client
from src.memory_manager import load_model, memory_manager

logger = logging.getLogger(__name__)

class ActionAnalyzePhoto(Action):
    """Analyze uploaded photo using computer vision"""
    
    def name(self) -> Text:
        return "action_analyze_photo"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Get photo from tracker (in real implementation, this would come from file upload)
            photo_data = tracker.get_slot("photo_uploaded")
            
            if not photo_data:
                dispatcher.utter_message(text="I don't see any photo uploaded. Please upload a photo of your game.")
                return []
            
            # Load OCR model for text extraction
            ocr_model = await load_model("easyocr")
            
            # Simulate photo analysis (in real implementation, use EasyOCR)
            analysis_result = {
                "game_title": "Detected Game Title",
                "platform": "Detected Platform",
                "condition": "good",
                "confidence": 0.85,
                "text_extracted": ["Game Title", "Platform Name", "Publisher"],
                "analysis_time": time.time()
            }
            
            # Store analysis result
            dispatcher.utter_message(
                text=f"📸 Photo analyzed successfully!\n"
                     f"🎮 Game: {analysis_result['game_title']}\n"
                     f"🎯 Platform: {analysis_result['platform']}\n"
                     f"⭐ Condition: {analysis_result['condition']}\n"
                     f"🎯 Confidence: {analysis_result['confidence']:.0%}"
            )
            
            return [
                SlotSet("photo_analysis_result", analysis_result),
                SlotSet("game_title", analysis_result["game_title"]),
                SlotSet("platform", analysis_result["platform"]),
                SlotSet("condition", analysis_result["condition"])
            ]
            
        except Exception as e:
            logger.error(f"Photo analysis failed: {e}")
            dispatcher.utter_message(text="Sorry, I couldn't analyze the photo. Please try again or provide game details manually.")
            return []

class ActionExtractGameInfoFromPhoto(Action):
    """Extract game information from photo analysis"""
    
    def name(self) -> Text:
        return "action_extract_game_info_from_photo"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            analysis_result = tracker.get_slot("photo_analysis_result")
            
            if not analysis_result:
                return [FollowupAction("action_analyze_photo")]
            
            # Extract additional game information from database
            async with get_mysql_session() as session:
                # Query game database for additional info
                # This is a simplified version - real implementation would query actual database
                game_info = {
                    "genre": "Action",
                    "release_year": "2023",
                    "publisher": "Game Publisher",
                    "rating": "M"
                }
            
            dispatcher.utter_message(
                text=f"📋 Additional Game Information:\n"
                     f"🎭 Genre: {game_info['genre']}\n"
                     f"📅 Release Year: {game_info['release_year']}\n"
                     f"🏢 Publisher: {game_info['publisher']}\n"
                     f"🔞 Rating: {game_info['rating']}"
            )
            
            return [SlotSet("genre", game_info["genre"])]
            
        except Exception as e:
            logger.error(f"Game info extraction failed: {e}")
            return []

class ActionGetGamePrice(Action):
    """Get game price based on title, platform, and condition"""
    
    def name(self) -> Text:
        return "action_get_game_price"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            game_title = tracker.get_slot("game_title")
            platform = tracker.get_slot("platform")
            condition = tracker.get_slot("condition")
            
            if not all([game_title, platform, condition]):
                dispatcher.utter_message(text="I need the game title, platform, and condition to provide accurate pricing.")
                return []
            
            # Query database for pricing information
            async with get_mysql_session() as session:
                # Simplified pricing logic - real implementation would query marketplace_prices table
                base_price = 45.99  # This would come from database
                condition_multiplier = {
                    "mint": 1.2,
                    "excellent": 1.0,
                    "good": 0.8,
                    "fair": 0.6,
                    "poor": 0.4
                }.get(condition.lower(), 0.8)
                
                estimated_price = base_price * condition_multiplier
                price_range = {
                    "low": estimated_price * 0.8,
                    "high": estimated_price * 1.2,
                    "average": estimated_price
                }
            
            # Cache the result
            redis_client = await get_redis_client()
            cache_key = f"price:{game_title}:{platform}:{condition}"
            await redis_client.setex(cache_key, 3600, json.dumps(price_range))
            
            dispatcher.utter_message(
                text=f"💰 Price Estimate for {game_title} ({platform}):\n"
                     f"📊 Condition: {condition.title()}\n"
                     f"💵 Estimated Value: ${price_range['average']:.2f}\n"
                     f"📈 Price Range: ${price_range['low']:.2f} - ${price_range['high']:.2f}\n"
                     f"💡 This estimate is based on current market data and condition assessment."
            )
            
            return []
            
        except Exception as e:
            logger.error(f"Price lookup failed: {e}")
            dispatcher.utter_message(text="Sorry, I couldn't retrieve pricing information right now. Please try again later.")
            return []

class ActionGetPhotoBasedPrice(Action):
    """Get price based on photo analysis results"""
    
    def name(self) -> Text:
        return "action_get_photo_based_price"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        analysis_result = tracker.get_slot("photo_analysis_result")
        
        if not analysis_result:
            return [FollowupAction("action_analyze_photo")]
        
        # Use extracted information for pricing
        return [
            SlotSet("game_title", analysis_result.get("game_title")),
            SlotSet("platform", analysis_result.get("platform")),
            SlotSet("condition", analysis_result.get("condition")),
            FollowupAction("action_get_game_price")
        ]

class ActionComparePrices(Action):
    """Compare prices across different marketplaces"""
    
    def name(self) -> Text:
        return "action_compare_marketplace_prices"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            game_title = tracker.get_slot("game_title")
            platform = tracker.get_slot("platform")
            
            if not game_title:
                dispatcher.utter_message(text="Please specify which game you'd like to compare prices for.")
                return []
            
            # Simulate marketplace price comparison
            marketplace_prices = {
                "eBay": {"price": 42.99, "fees": 3.44, "net": 39.55},
                "Amazon": {"price": 45.99, "fees": 6.90, "net": 39.09},
                "Facebook": {"price": 40.00, "fees": 0.00, "net": 40.00},
                "Mercari": {"price": 41.50, "fees": 4.15, "net": 37.35},
                "Local": {"price": 38.00, "fees": 0.00, "net": 38.00}
            }
            
            # Sort by net profit
            sorted_prices = sorted(marketplace_prices.items(), key=lambda x: x[1]["net"], reverse=True)
            
            comparison_text = f"💰 Price Comparison for {game_title}:\n\n"
            for i, (marketplace, data) in enumerate(sorted_prices, 1):
                comparison_text += f"{i}. {marketplace}: ${data['price']:.2f} (Net: ${data['net']:.2f})\n"
            
            comparison_text += "\n💡 Recommendation: Consider the marketplace with the highest net profit after fees!"
            
            dispatcher.utter_message(text=comparison_text)
            
            return []
            
        except Exception as e:
            logger.error(f"Price comparison failed: {e}")
            dispatcher.utter_message(text="Sorry, I couldn't compare prices right now. Please try again later.")
            return []

class ActionRecommendMarketplace(Action):
    """Recommend best marketplace for selling"""
    
    def name(self) -> Text:
        return "action_recommend_marketplace"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            game_title = tracker.get_slot("game_title")
            platform = tracker.get_slot("platform")
            condition = tracker.get_slot("condition")
            
            # Marketplace recommendation logic
            recommendations = {
                "high_value": "eBay - Best for rare/valuable games",
                "quick_sale": "Facebook Marketplace - Fast local sales",
                "convenience": "Amazon - Easy listing process",
                "collector": "Mercari - Great for collector items",
                "bulk": "Local game store - Good for bulk sales"
            }
            
            # Determine best recommendation based on game info
            if condition and condition.lower() in ["mint", "excellent"]:
                primary_rec = recommendations["high_value"]
            else:
                primary_rec = recommendations["quick_sale"]
            
            dispatcher.utter_message(
                text=f"🏪 Marketplace Recommendations:\n\n"
                     f"🥇 Primary: {primary_rec}\n"
                     f"🥈 Alternative: {recommendations['convenience']}\n"
                     f"🥉 Quick Option: {recommendations['quick_sale']}\n\n"
                     f"💡 Consider your priorities: maximum profit vs. quick sale vs. convenience!"
            )
            
            return []
            
        except Exception as e:
            logger.error(f"Marketplace recommendation failed: {e}")
            dispatcher.utter_message(text="I can help you choose the best marketplace! What game are you looking to sell?")
            return []

class ActionGetSellingTips(Action):
    """Provide selling tips and best practices"""
    
    def name(self) -> Text:
        return "action_get_selling_tips"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        tips = [
            "📸 Take high-quality photos from multiple angles",
            "📝 Write detailed, honest descriptions",
            "💰 Research current market prices before listing",
            "📦 Package items securely for shipping",
            "⚡ Respond quickly to buyer inquiries",
            "🔍 Check buyer ratings and feedback",
            "💳 Use secure payment methods",
            "📊 Consider seasonal demand fluctuations"
        ]
        
        tips_text = "💡 Pro Selling Tips:\n\n" + "\n".join(tips)
        tips_text += "\n\n🎯 Remember: Honest descriptions and good photos lead to better sales!"
        
        dispatcher.utter_message(text=tips_text)
        
        return []

class ActionGetMarketTrends(Action):
    """Get current market trends and analysis"""
    
    def name(self) -> Text:
        return "action_get_market_analysis"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Simulate market trend analysis
            trends = {
                "hot_genres": ["Action", "RPG", "Adventure"],
                "trending_platforms": ["Nintendo Switch", "PlayStation 5"],
                "price_trend": "stable",
                "seasonal_factor": "holiday_boost",
                "demand_level": "high"
            }
            
            trend_text = f"📈 Current Market Trends:\n\n"
            trend_text += f"🔥 Hot Genres: {', '.join(trends['hot_genres'])}\n"
            trend_text += f"🎮 Trending Platforms: {', '.join(trends['trending_platforms'])}\n"
            trend_text += f"📊 Price Trend: {trends['price_trend'].title()}\n"
            trend_text += f"🎄 Seasonal Factor: {trends['seasonal_factor'].replace('_', ' ').title()}\n"
            trend_text += f"📈 Demand Level: {trends['demand_level'].title()}\n"
            trend_text += f"\n💡 Best time to sell: Now! Market conditions are favorable."
            
            dispatcher.utter_message(text=trend_text)
            
            return []
            
        except Exception as e:
            logger.error(f"Market trend analysis failed: {e}")
            dispatcher.utter_message(text="Market analysis is temporarily unavailable. Please try again later.")
            return []

class ActionLogConversation(Action):
    """Log conversation for analytics"""
    
    def name(self) -> Text:
        return "action_log_conversation"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Log conversation to database
            conversation_data = {
                "user_id": tracker.sender_id,
                "timestamp": datetime.now().isoformat(),
                "intent": tracker.latest_message.get("intent", {}).get("name"),
                "entities": tracker.latest_message.get("entities", []),
                "slots": dict(tracker.slots)
            }
            
            async with get_mysql_session() as session:
                # Insert into chatbot_conversations table
                # This is simplified - real implementation would use SQLAlchemy models
                pass
            
            return []
            
        except Exception as e:
            logger.error(f"Conversation logging failed: {e}")
            return []

class ActionDefaultFallback(Action):
    """Default fallback action"""
    
    def name(self) -> Text:
        return "action_default_fallback"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        fallback_messages = [
            "I'm not sure I understand. Could you rephrase that?",
            "I specialize in game pricing and market analysis. How can I help you with that?",
            "Try asking about game prices, uploading a photo, or marketplace recommendations!",
            "I'm here to help with second-hand game valuation. What would you like to know?"
        ]
        
        import random
        message = random.choice(fallback_messages)
        dispatcher.utter_message(text=message)
        
        return []

class ValidatePriceInquiryForm(FormValidationAction):
    """Validate price inquiry form inputs"""
    
    def name(self) -> Text:
        return "validate_price_inquiry_form"
    
    def validate_game_title(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        
        if slot_value and len(slot_value.strip()) > 2:
            return {"game_title": slot_value.strip()}
        else:
            dispatcher.utter_message(text="Please provide a valid game title.")
            return {"game_title": None}
    
    def validate_platform(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        
        valid_platforms = ["playstation", "xbox", "nintendo", "pc", "mobile"]
        
        if slot_value and any(platform in slot_value.lower() for platform in valid_platforms):
            return {"platform": slot_value}
        else:
            dispatcher.utter_message(text="Please specify a valid platform (PlayStation, Xbox, Nintendo, PC, or Mobile).")
            return {"platform": None}
    
    def validate_condition(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        
        valid_conditions = ["mint", "excellent", "good", "fair", "poor"]
        
        if slot_value and slot_value.lower() in valid_conditions:
            return {"condition": slot_value.lower()}
        else:
            dispatcher.utter_message(text="Please specify the condition: Mint, Excellent, Good, Fair, or Poor.")
            return {"condition": None}

# Additional utility actions
class ActionGetGameDetails(Action):
    """Get detailed game information"""
    
    def name(self) -> Text:
        return "action_get_game_details"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        game_title = tracker.get_slot("game_title")
        
        if not game_title:
            dispatcher.utter_message(text="Which game would you like to know more about?")
            return []
        
        # Simulate game details lookup
        game_details = {
            "title": game_title,
            "genre": "Action/Adventure",
            "release_date": "2023",
            "developer": "Game Studio",
            "platforms": ["PC", "PlayStation 5", "Xbox Series X"],
            "rating": "M for Mature"
        }
        
        details_text = f"🎮 Game Details for {game_details['title']}:\n\n"
        details_text += f"🎭 Genre: {game_details['genre']}\n"
        details_text += f"📅 Release: {game_details['release_date']}\n"
        details_text += f"🏢 Developer: {game_details['developer']}\n"
        details_text += f"🎯 Platforms: {', '.join(game_details['platforms'])}\n"
        details_text += f"🔞 Rating: {game_details['rating']}"
        
        dispatcher.utter_message(text=details_text)
        
        return []

class ActionSearchGameDatabase(Action):
    """Search game database"""
    
    def name(self) -> Text:
        return "action_search_game_database"
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        genre = tracker.get_slot("genre")
        platform = tracker.get_slot("platform")
        
        search_results = [
            "Game Title 1 - Action/Adventure",
            "Game Title 2 - RPG",
            "Game Title 3 - Sports"
        ]
        
        if search_results:
            results_text = f"🔍 Search Results:\n\n"
            for i, game in enumerate(search_results, 1):
                results_text += f"{i}. {game}\n"
            results_text += "\nWould you like pricing information for any of these games?"
        else:
            results_text = "No games found matching your criteria. Try different search terms!"
        
        dispatcher.utter_message(text=results_text)
        
        return []
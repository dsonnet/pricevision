"""
PriceVision Photo Analysis Module
RTX 3050 optimized computer vision for game identification
"""

import logging
import asyncio
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available, using mock OCR")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

from .config import config
from .memory_manager import memory_manager, ModelPriority
from .database import get_redis_client, get_mysql_session

logger = logging.getLogger(__name__)

class RTX3050PhotoAnalyzer:
    """
    Photo analysis system optimized for RTX 3050 (4GB VRAM)
    Uses CPU-based OCR to preserve GPU memory for other models
    """
    
    def __init__(self):
        self.ocr_reader = None
        self.game_patterns = []
        self.platform_patterns = []
        self.condition_keywords = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # RTX 3050 optimizations
        self.max_image_size = (1024, 1024)  # Limit image size for memory
        self.ocr_batch_size = 1  # Process one image at a time
        self.use_cpu_only = True  # Force CPU for OCR to save GPU memory
        
    async def initialize(self):
        """Initialize the photo analyzer with RTX 3050 optimizations"""
        try:
            logger.info("Initializing RTX 3050 optimized photo analyzer...")
            
            # Initialize OCR reader (CPU-only for RTX 3050)
            await self._setup_ocr_reader()
            
            # Load game recognition patterns
            await self._load_recognition_patterns()
            
            logger.info("Photo analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize photo analyzer: {e}")
            raise
    
    async def _setup_ocr_reader(self):
        """Setup EasyOCR reader with CPU-only mode"""
        try:
            if EASYOCR_AVAILABLE:
                # Force CPU usage to preserve GPU memory for other models
                self.ocr_reader = easyocr.Reader(
                    ['en'],  # English only for better performance
                    gpu=False,  # Force CPU usage for RTX 3050
                    verbose=False,
                    download_enabled=True
                )
                logger.info("EasyOCR initialized (CPU mode)")
            else:
                self.ocr_reader = MockOCRReader()
                logger.info("Using mock OCR reader")
                
        except Exception as e:
            logger.error(f"Failed to setup OCR reader: {e}")
            self.ocr_reader = MockOCRReader()
    
    async def _load_recognition_patterns(self):
        """Load game and platform recognition patterns"""
        try:
            # Game title patterns (common words/phrases in game titles)
            self.game_patterns = [
                r'(?i)\b(call of duty|cod)\b',
                r'(?i)\b(grand theft auto|gta)\b',
                r'(?i)\b(the legend of zelda|zelda)\b',
                r'(?i)\b(super mario|mario)\b',
                r'(?i)\b(pokemon|pokémon)\b',
                r'(?i)\b(final fantasy|ff)\b',
                r'(?i)\b(assassin\'?s creed)\b',
                r'(?i)\b(red dead redemption)\b',
                r'(?i)\b(god of war)\b',
                r'(?i)\b(the witcher)\b',
                r'(?i)\b(cyberpunk)\b',
                r'(?i)\b(minecraft)\b',
                r'(?i)\b(fortnite)\b',
                r'(?i)\b(apex legends)\b',
                r'(?i)\b(overwatch)\b'
            ]
            
            # Platform patterns
            self.platform_patterns = [
                r'(?i)\b(playstation|ps[1-5]|psx|ps vita|psp)\b',
                r'(?i)\b(xbox|xbox one|xbox 360|xbox series [xs])\b',
                r'(?i)\b(nintendo|switch|3ds|ds|wii|gamecube|n64)\b',
                r'(?i)\b(pc|steam|epic games|origin|uplay)\b',
                r'(?i)\b(mobile|ios|android|app store|google play)\b'
            ]
            
            # Condition assessment keywords
            self.condition_keywords = {
                'mint': ['mint', 'perfect', 'flawless', 'pristine', 'new'],
                'excellent': ['excellent', 'great', 'very good', 'like new'],
                'good': ['good', 'decent', 'working', 'functional'],
                'fair': ['fair', 'okay', 'some wear', 'used'],
                'poor': ['poor', 'damaged', 'broken', 'cracked', 'scratched']
            }
            
            logger.info("Recognition patterns loaded")
            
        except Exception as e:
            logger.error(f"Failed to load recognition patterns: {e}")
    
    async def analyze_photo(self, image_data: bytes, filename: str = "image") -> Dict[str, Any]:
        """
        Analyze uploaded photo to extract game information
        
        Args:
            image_data: Raw image bytes
            filename: Original filename for context
            
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"photo_analysis:{hash(image_data)}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached photo analysis result")
                return cached_result
            
            # Preprocess image
            processed_image = await self._preprocess_image(image_data)
            
            # Extract text using OCR
            ocr_results = await self._extract_text(processed_image)
            
            # Analyze extracted text
            analysis_result = await self._analyze_extracted_text(ocr_results)
            
            # Assess image quality and condition
            quality_assessment = await self._assess_image_quality(processed_image)
            
            # Combine results
            final_result = {
                "success": True,
                "filename": filename,
                "analysis_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "ocr_results": ocr_results,
                "game_info": analysis_result,
                "quality_assessment": quality_assessment,
                "confidence_score": self._calculate_confidence(analysis_result, quality_assessment),
                "recommendations": self._generate_recommendations(analysis_result, quality_assessment)
            }
            
            # Cache the result
            await self._cache_result(cache_key, final_result)
            
            logger.info(f"Photo analysis completed in {final_result['analysis_time']:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Photo analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "filename": filename
            }
    
    async def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (RTX 3050 memory optimization)
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Enhance image for better OCR
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert to grayscale for OCR
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding for better text detection
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original image as fallback
            image = Image.open(io.BytesIO(image_data))
            return np.array(image.convert('RGB'))
    
    async def _extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            if not self.ocr_reader:
                await self._setup_ocr_reader()
            
            # Use memory management for OCR operation
            async with memory_manager.load_model_context(
                "ocr",
                ModelPriority.MEDIUM,
                estimated_memory_mb=150
            ):
                if EASYOCR_AVAILABLE and hasattr(self.ocr_reader, 'readtext'):
                    # EasyOCR processing
                    results = self.ocr_reader.readtext(
                        image,
                        detail=1,  # Return bounding boxes and confidence
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7
                    )
                    
                    # Process results
                    extracted_text = []
                    all_text = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:  # Filter low-confidence results
                            extracted_text.append({
                                "text": text.strip(),
                                "confidence": confidence,
                                "bbox": bbox
                            })
                            all_text.append(text.strip())
                    
                    return {
                        "raw_results": results,
                        "extracted_text": extracted_text,
                        "all_text": " ".join(all_text),
                        "text_count": len(extracted_text)
                    }
                else:
                    # Mock OCR results
                    return self.ocr_reader.readtext(image)
                    
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "extracted_text": [],
                "all_text": "",
                "text_count": 0,
                "error": str(e)
            }
    
    async def _analyze_extracted_text(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze extracted text to identify game information"""
        try:
            all_text = ocr_results.get("all_text", "").lower()
            extracted_items = ocr_results.get("extracted_text", [])
            
            analysis = {
                "game_title": None,
                "platform": None,
                "genre": None,
                "publisher": None,
                "rating": None,
                "year": None,
                "edition": None,
                "matches": []
            }
            
            # Game title detection
            import re
            for pattern in self.game_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    analysis["game_title"] = matches[0]
                    analysis["matches"].append(f"Game: {matches[0]}")
                    break
            
            # Platform detection
            for pattern in self.platform_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    analysis["platform"] = matches[0]
                    analysis["matches"].append(f"Platform: {matches[0]}")
                    break
            
            # Look for additional information
            # ESRB ratings
            rating_pattern = r'(?i)\b(e|t|m|ao|rp|e10\+|everyone|teen|mature)\b'
            rating_matches = re.findall(rating_pattern, all_text)
            if rating_matches:
                analysis["rating"] = rating_matches[0].upper()
            
            # Year detection
            year_pattern = r'\b(19|20)\d{2}\b'
            year_matches = re.findall(year_pattern, all_text)
            if year_matches:
                analysis["year"] = year_matches[0]
            
            # Edition detection
            edition_pattern = r'(?i)\b(collector\'?s|limited|special|deluxe|premium|standard|goty|game of the year)\b'
            edition_matches = re.findall(edition_pattern, all_text)
            if edition_matches:
                analysis["edition"] = edition_matches[0]
            
            # Publisher detection (common publishers)
            publisher_pattern = r'(?i)\b(nintendo|sony|microsoft|activision|ubisoft|ea|electronic arts|bethesda|rockstar|square enix|capcom|konami|sega|bandai namco)\b'
            publisher_matches = re.findall(publisher_pattern, all_text)
            if publisher_matches:
                analysis["publisher"] = publisher_matches[0]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {"error": str(e)}
    
    async def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality and condition indicators"""
        try:
            assessment = {
                "image_quality": "good",
                "condition_indicators": [],
                "clarity_score": 0.0,
                "lighting_score": 0.0,
                "overall_score": 0.0
            }
            
            # Calculate image sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            assessment["clarity_score"] = min(laplacian_var / 1000, 1.0)
            
            # Calculate lighting quality (histogram analysis)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            lighting_score = 1.0 - (np.std(hist) / np.mean(hist)) if np.mean(hist) > 0 else 0.0
            assessment["lighting_score"] = max(0.0, min(lighting_score, 1.0))
            
            # Overall quality score
            assessment["overall_score"] = (assessment["clarity_score"] + assessment["lighting_score"]) / 2
            
            # Quality classification
            if assessment["overall_score"] > 0.7:
                assessment["image_quality"] = "excellent"
            elif assessment["overall_score"] > 0.5:
                assessment["image_quality"] = "good"
            elif assessment["overall_score"] > 0.3:
                assessment["image_quality"] = "fair"
            else:
                assessment["image_quality"] = "poor"
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"image_quality": "unknown", "error": str(e)}
    
    def _calculate_confidence(self, analysis_result: Dict[str, Any], quality_assessment: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []
            
            # Game title confidence
            if analysis_result.get("game_title"):
                confidence_factors.append(0.8)
            
            # Platform confidence
            if analysis_result.get("platform"):
                confidence_factors.append(0.7)
            
            # Image quality factor
            quality_score = quality_assessment.get("overall_score", 0.5)
            confidence_factors.append(quality_score)
            
            # Number of matches factor
            match_count = len(analysis_result.get("matches", []))
            match_factor = min(match_count / 3, 1.0)
            confidence_factors.append(match_factor)
            
            # Calculate weighted average
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.3  # Low confidence if no factors
                
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any], quality_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for better results"""
        recommendations = []
        
        try:
            # Image quality recommendations
            if quality_assessment.get("image_quality") == "poor":
                recommendations.append("📸 Try taking a clearer photo with better lighting")
                recommendations.append("🔍 Ensure the game title and platform are clearly visible")
            
            # Missing information recommendations
            if not analysis_result.get("game_title"):
                recommendations.append("🎮 Make sure the game title is clearly visible in the photo")
            
            if not analysis_result.get("platform"):
                recommendations.append("🎯 Include the platform logo or text in the photo")
            
            # General recommendations
            recommendations.append("💡 For best results, photograph the front cover directly")
            recommendations.append("📱 Avoid reflections and shadows on the game case")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["📸 Try taking a clearer photo for better analysis"]
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        try:
            redis_client = await get_redis_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result"""
        try:
            redis_client = await get_redis_client()
            await redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result, default=str)
            )
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    async def batch_analyze_photos(self, image_list: List[Tuple[bytes, str]]) -> List[Dict[str, Any]]:
        """Analyze multiple photos in batch (memory-optimized for RTX 3050)"""
        results = []
        
        try:
            # Process one at a time to manage memory
            for i, (image_data, filename) in enumerate(image_list):
                logger.info(f"Processing image {i+1}/{len(image_list)}: {filename}")
                
                result = await self.analyze_photo(image_data, filename)
                results.append(result)
                
                # Memory cleanup between images
                await memory_manager.cleanup_if_needed()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return results
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.ocr_reader = None
            await memory_manager.cleanup_if_needed()
            logger.info("Photo analyzer cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

class MockOCRReader:
    """Mock OCR reader for development without EasyOCR"""
    
    def readtext(self, image):
        """Mock OCR text extraction"""
        mock_results = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Call of Duty", 0.9),
            ([[0, 40], [80, 40], [80, 60], [0, 60]], "PlayStation 5", 0.8),
            ([[0, 70], [60, 70], [60, 90], [0, 90]], "2023", 0.7)
        ]
        
        return {
            "raw_results": mock_results,
            "extracted_text": [
                {"text": "Call of Duty", "confidence": 0.9, "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
                {"text": "PlayStation 5", "confidence": 0.8, "bbox": [[0, 40], [80, 40], [80, 60], [0, 60]]},
                {"text": "2023", "confidence": 0.7, "bbox": [[0, 70], [60, 70], [60, 90], [0, 90]]}
            ],
            "all_text": "Call of Duty PlayStation 5 2023",
            "text_count": 3
        }

# Global photo analyzer instance
photo_analyzer = RTX3050PhotoAnalyzer()

async def initialize_photo_analyzer():
    """Initialize the global photo analyzer"""
    await photo_analyzer.initialize()

async def analyze_game_photo(image_data: bytes, filename: str = "image") -> Dict[str, Any]:
    """Analyze a game photo"""
    if not photo_analyzer.ocr_reader:
        await initialize_photo_analyzer()
    
    return await photo_analyzer.analyze_photo(image_data, filename)

async def batch_analyze_game_photos(image_list: List[Tuple[bytes, str]]) -> List[Dict[str, Any]]:
    """Analyze multiple game photos"""
    if not photo_analyzer.ocr_reader:
        await initialize_photo_analyzer()
    
    return await photo_analyzer.batch_analyze_photos(image_list)

# Context manager for photo analysis operations
@asynccontextmanager
async def photo_analysis_context():
    """Context manager for photo analysis operations"""
    try:
        if not photo_analyzer.ocr_reader:
            await initialize_photo_analyzer()
        yield photo_analyzer
    finally:
        await memory_manager.cleanup_if_needed()
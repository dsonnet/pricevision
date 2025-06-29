#!/usr/bin/env python3
"""
PriceVision Main Application Entry Point
RTX 3050 Optimized Implementation

This is the main entry point for the PriceVision chatbot system.
It initializes all components and starts the FastAPI server.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config, PriceVisionConfig, RTX3050Config
from database import initialize_database, DatabaseManager
from memory_manager import initialize_memory_manager, memory_manager
from rag_system import initialize_rag_system
from photo_analysis import initialize_photo_analyzer
from main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pricevision.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class PriceVisionApp:
    """Main application class for PriceVision"""
    
    def __init__(self):
        self.settings = config
        self.rtx3050_config = config.rtx3050
        self.db_manager: Optional[DatabaseManager] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all application components"""
        try:
            logger.info("🚀 Starting PriceVision initialization...")
            
            # Create logs directory
            os.makedirs('logs', exist_ok=True)
            
            # 1. Initialize memory manager first (critical for RTX 3050)
            logger.info("📊 Initializing memory manager...")
            await initialize_memory_manager()
            
            # 2. Initialize database connections
            logger.info("🗄️ Initializing database...")
            await initialize_database()
            
            # 3. Initialize RAG system
            logger.info("🧠 Initializing RAG system...")
            await initialize_rag_system()
            
            # 4. Initialize photo analyzer
            logger.info("📸 Initializing photo analyzer...")
            await initialize_photo_analyzer()
            
            # 5. Verify GPU status
            await self._verify_gpu_setup()
            
            self.initialized = True
            logger.info("✅ PriceVision initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _verify_gpu_setup(self):
        """Verify GPU setup and memory constraints"""
        try:
            gpu_status = await memory_manager.get_gpu_status()
            
            if gpu_status.get('available', False):
                gpu_name = gpu_status.get('name', 'Unknown')
                total_memory = gpu_status.get('total_memory_mb', 0)
                
                logger.info(f"🎮 GPU detected: {gpu_name}")
                logger.info(f"💾 Total GPU memory: {total_memory:.0f} MB")
                
                if 'RTX 3050' in gpu_name or total_memory <= 4096:
                    logger.info("⚡ RTX 3050 optimizations active")
                    logger.info(f"🔒 Memory limit: {self.rtx3050_config.max_gpu_memory_mb} MB")
                else:
                    logger.warning("⚠️ GPU detected but not RTX 3050 - using conservative settings")
            else:
                logger.warning("⚠️ No GPU detected - running in CPU mode")
                
        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        if not self.initialized:
            await self.initialize()
        
        try:
            import uvicorn
            
            logger.info(f"🌐 Starting PriceVision server on {host}:{port}")
            logger.info("📋 Available endpoints:")
            logger.info("  • GET  /health - Health check")
            logger.info("  • POST /chat - Chat with the bot")
            logger.info("  • POST /analyze-photo - Analyze game photos")
            logger.info("  • POST /price-inquiry - Get price estimates")
            logger.info("  • GET  /system/status - System monitoring")
            
            # Configure uvicorn for RTX 3050 constraints
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                workers=1,  # Single worker for memory constraints
                log_level="info",
                access_log=True,
                reload=False  # Disable reload in production
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("🛑 Shutting down PriceVision...")
            
            # Cleanup memory manager
            if memory_manager:
                await memory_manager.cleanup_all()
            
            # Close database connections
            if self.db_manager:
                await self.db_manager.close()
            
            logger.info("✅ Shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

async def main():
    """Main entry point"""
    app_instance = PriceVisionApp()
    
    try:
        # Handle command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='PriceVision Chatbot Server')
        parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
        parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
        parser.add_argument('--init-only', action='store_true', help='Initialize only, don\'t start server')
        
        args = parser.parse_args()
        
        if args.init_only:
            # Initialize only mode for testing
            await app_instance.initialize()
            logger.info("🔧 Initialization complete - exiting (init-only mode)")
        else:
            # Normal server mode
            await app_instance.start_server(args.host, args.port)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)
    finally:
        await app_instance.shutdown()

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
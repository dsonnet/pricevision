#!/usr/bin/env python3
"""
PriceVision System Startup Script
RTX 3050 Optimized

This script starts the complete PriceVision system with proper initialization.
"""

import asyncio
import sys
import os
import signal
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages the complete PriceVision system"""
    
    def __init__(self):
        self.app_instance = None
        self.shutdown_event = asyncio.Event()
        
    async def start_system(self):
        """Start the complete system"""
        try:
            logger.info("🚀 Starting PriceVision System...")
            
            # Import and initialize the main app
            from app import PriceVisionApp
            
            self.app_instance = PriceVisionApp()
            
            # Initialize all components
            await self.app_instance.initialize()
            
            # Start the FastAPI server
            logger.info("🌐 Starting FastAPI server...")
            await self.app_instance.start_server()
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
            await self.shutdown()
        except Exception as e:
            logger.error(f"💥 System startup failed: {e}")
            await self.shutdown()
            sys.exit(1)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🔄 Shutting down system...")
        
        if self.app_instance:
            await self.app_instance.shutdown()
        
        self.shutdown_event.set()
        logger.info("✅ System shutdown complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    system_manager = SystemManager()
    system_manager.setup_signal_handlers()
    
    try:
        await system_manager.start_system()
        
        # Wait for shutdown signal
        await system_manager.shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"💥 System crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("src/app.py").exists():
        print("❌ Please run from the project root directory")
        sys.exit(1)
    
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
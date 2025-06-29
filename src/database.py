"""
PriceVision Database Module
Handles MySQL, Redis, and ChromaDB connections
"""

import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import logging

# Database imports
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
import redis.asyncio as redis
import chromadb
from chromadb.config import Settings

# Local imports
from .config import config

# Setup logging
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

class DatabaseManager:
    """Manages all database connections for PriceVision"""
    
    def __init__(self):
        self.mysql_engine = None
        self.mysql_async_engine = None
        self.mysql_session_factory = None
        self.mysql_async_session_factory = None
        self.redis_client = None
        self.chroma_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all database connections"""
        if self._initialized:
            return
        
        try:
            # Initialize MySQL
            await self._init_mysql()
            
            # Initialize Redis
            await self._init_redis()
            
            # Initialize ChromaDB
            await self._init_chromadb()
            
            self._initialized = True
            logger.info("✅ All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _init_mysql(self):
        """Initialize MySQL connections"""
        try:
            # Synchronous engine for migrations and setup
            self.mysql_engine = create_engine(
                config.database.mysql_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=config.development_mode
            )
            
            # Async engine for application use
            async_url = config.database.mysql_url.replace("mysql+pymysql://", "mysql+aiomysql://")
            self.mysql_async_engine = create_async_engine(
                async_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=config.development_mode
            )
            
            # Session factories
            self.mysql_session_factory = sessionmaker(bind=self.mysql_engine)
            self.mysql_async_session_factory = async_sessionmaker(
                bind=self.mysql_async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("✅ MySQL connection initialized")
            
        except Exception as e:
            logger.error(f"❌ MySQL initialization failed: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                config.database.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            raise
    
    async def _init_chromadb(self):
        """Initialize ChromaDB connection"""
        try:
            self.chroma_client = chromadb.HttpClient(
                host=config.database.chromadb_host,
                port=config.database.chromadb_port,
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                    chroma_client_auth_credentials_provider="chromadb.auth.basic.BasicAuthCredentialsProvider"
                )
            )
            
            # Test connection by listing collections
            collections = self.chroma_client.list_collections()
            logger.info(f"✅ ChromaDB connection initialized ({len(collections)} collections)")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            raise
    
    @asynccontextmanager
    async def get_mysql_session(self):
        """Get async MySQL session context manager"""
        if not self.mysql_async_session_factory:
            raise RuntimeError("MySQL not initialized")
        
        async with self.mysql_async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_mysql_sync_session(self):
        """Get synchronous MySQL session"""
        if not self.mysql_session_factory:
            raise RuntimeError("MySQL not initialized")
        return self.mysql_session_factory()
    
    async def get_redis_client(self):
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        return self.redis_client
    
    def get_chroma_client(self):
        """Get ChromaDB client"""
        if not self.chroma_client:
            raise RuntimeError("ChromaDB not initialized")
        return self.chroma_client
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.mysql_async_engine:
                await self.mysql_async_engine.dispose()
            
            if self.mysql_engine:
                self.mysql_engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ All database connections closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {
            "mysql": False,
            "redis": False,
            "chromadb": False
        }
        
        # Check MySQL
        try:
            async with self.get_mysql_session() as session:
                await session.execute("SELECT 1")
            health["mysql"] = True
        except Exception as e:
            logger.error(f"MySQL health check failed: {e}")
        
        # Check Redis
        try:
            redis_client = await self.get_redis_client()
            await redis_client.ping()
            health["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check ChromaDB
        try:
            chroma_client = self.get_chroma_client()
            chroma_client.heartbeat()
            health["chromadb"] = True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
        
        return health

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
async def get_mysql_session():
    """Get MySQL session - convenience function"""
    return db_manager.get_mysql_session()

async def get_redis_client():
    """Get Redis client - convenience function"""
    return await db_manager.get_redis_client()

def get_chroma_client():
    """Get ChromaDB client - convenience function"""
    return db_manager.get_chroma_client()

class CacheManager:
    """Redis-based cache manager for RTX 3050 optimization"""
    
    def __init__(self):
        self.redis_client = None
    
    async def initialize(self):
        """Initialize cache manager"""
        self.redis_client = await get_redis_client()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if not self.redis_client:
                await self.initialize()
            
            value = await self.redis_client.get(key)
            if value:
                import json
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            if not self.redis_client:
                await self.initialize()
            
            import json
            await self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            if not self.redis_client:
                await self.initialize()
            
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        try:
            if not self.redis_client:
                await self.initialize()
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")

# Global cache manager
cache_manager = CacheManager()

# Database initialization function
async def initialize_database():
    """Initialize all databases - main entry point"""
    logger.info("🚀 Initializing PriceVision Database System")
    await db_manager.initialize()
    await cache_manager.initialize()
    logger.info("✅ Database system initialized successfully")

async def init_databases():
    """Initialize all databases"""
    await db_manager.initialize()
    await cache_manager.initialize()

# Database cleanup function
async def close_databases():
    """Close all database connections"""
    await db_manager.close()
import os
from typing import Optional
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from pymongo import MongoClient
from redis import Redis
from pydantic import BaseSettings

class DatabaseSettings(BaseSettings):
    """Database connection settings"""
    # PostgreSQL settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "smart_shopping"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_POOL_MIN: int = 1
    POSTGRES_POOL_MAX: int = 20

    # MongoDB settings
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "smart_shopping"

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    class Config:
        env_file = ".env"

class DatabaseConnector:
    """Database connector for PostgreSQL, MongoDB, and Redis"""
    _instance = None
    _settings = DatabaseSettings()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize database connections"""
        # Initialize PostgreSQL connection pool
        self.pg_pool = SimpleConnectionPool(
            self._settings.POSTGRES_POOL_MIN,
            self._settings.POSTGRES_POOL_MAX,
            host=self._settings.POSTGRES_HOST,
            port=self._settings.POSTGRES_PORT,
            dbname=self._settings.POSTGRES_DB,
            user=self._settings.POSTGRES_USER,
            password=self._settings.POSTGRES_PASSWORD
        )

        # Initialize MongoDB client
        self.mongo_client = MongoClient(self._settings.MONGO_URI)
        self.mongo_db = self.mongo_client[self._settings.MONGO_DB]

        # Initialize Redis client
        self.redis_client = Redis(
            host=self._settings.REDIS_HOST,
            port=self._settings.REDIS_PORT,
            db=self._settings.REDIS_DB,
            password=self._settings.REDIS_PASSWORD,
            decode_responses=True
        )

    @contextmanager
    def get_postgres_connection(self):
        """Get a PostgreSQL connection from the pool"""
        conn = self.pg_pool.getconn()
        try:
            yield conn
        finally:
            self.pg_pool.putconn(conn)

    def get_mongo_collection(self, collection_name: str):
        """Get a MongoDB collection"""
        return self.mongo_db[collection_name]

    def get_redis_client(self) -> Redis:
        """Get Redis client"""
        return self.redis_client

    def close_all(self):
        """Close all database connections"""
        if hasattr(self, 'pg_pool'):
            self.pg_pool.closeall()
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
        if hasattr(self, 'redis_client'):
            self.redis_client.close()

# Example usage:
"""
db = DatabaseConnector()

# PostgreSQL
with db.get_postgres_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users")
        users = cur.fetchall()

# MongoDB
users_collection = db.get_mongo_collection('users')
user = users_collection.find_one({"user_id": "123"})

# Redis
redis_client = db.get_redis_client()
redis_client.set('user:123:session', 'active')
"""
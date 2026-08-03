from sqlalchemy import Table, Column, Integer, String, DateTime, func
from pgvector.sqlalchemy import Vector  
from db import metadata


EMBEDDING_DIM = 384

destinations = Table(
    "destinations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(100), nullable=False),
    Column("country", String(100), nullable=False),
    Column("description", String, nullable=False),
    Column("category", String(50), nullable=False),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=False)
)

search_history = Table(
    "search_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(50), nullable=False, default="default_user"),
    Column("query_text", String, nullable=False),
    Column("category_filter", String(50), nullable=True),
    Column("query_embedding", Vector(384), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)
from sqlalchemy import Table, Column, Integer, String
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
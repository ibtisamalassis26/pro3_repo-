import os
from sqlalchemy import create_engine, MetaData

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:123456@localhost:5432/travel_rec_db",
)

engine = create_engine(DATABASE_URL, echo=False)
metadata = MetaData()
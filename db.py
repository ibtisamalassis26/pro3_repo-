import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData

# Load variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/postgres")

engine = create_engine(DATABASE_URL)
metadata = MetaData()
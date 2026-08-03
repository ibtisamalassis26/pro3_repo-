import random
from sqlalchemy import select, func, insert
from db import engine, metadata
from models import destinations
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector  # 👈 Added pgvector import

# Load model once in memory
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

destinations_data = [
    {"name": "Paris", "country": "France", "description": "Famous for romantic vibes, iconic art museums, historical landmarks, and world-class culinary experiences.", "category": "Culture"},
    {"name": "Kyoto", "country": "Japan", "description": "Famed for classical Buddhist temples, gardens, imperial palaces, traditional wooden houses, and cherry blossoms.", "category": "Culture"},
    {"name": "Rome", "country": "Italy", "description": "Ancient ruins, Colosseum, Vatican City, rich history, cobblestone streets, and amazing pasta.", "category": "Culture"},
    {"name": "Cairo", "country": "Egypt", "description": "Home to the Giza Pyramids, ancient Sphinx, Nile river cruises, and vibrant bustling bazaars.", "category": "Culture"},
    {"name": "Bali", "country": "Indonesia", "description": "Tropical paradise featuring serene beaches, volcanic mountains, iconic rice terraces, and spiritual retreats.", "category": "Beach"},
    {"name": "Maldives", "country": "Maldives", "description": "Luxury overwater bungalows, crystal clear turquoise waters, vibrant coral reefs, and romantic sunsets.", "category": "Beach"},
    {"name": "Cancun", "country": "Mexico", "description": "White sand Caribbean beaches, lively nightlife resorts, and proximity to Mayan archaeological sites.", "category": "Beach"},
    {"name": "Phuket", "country": "Thailand", "description": "Exotic islands, energetic nightlife, colorful markets, and clear tropical waters perfect for diving.", "category": "Beach"},
    {"name": "Yellowstone", "country": "USA", "description": "Sprawling wilderness with dramatic geysers, hot springs, deep canyons, and abundant wildlife like bison.", "category": "Nature"},
    {"name": "Banff National Park", "country": "Canada", "description": "Turquoise glacial lakes, majestic Rocky Mountain peaks, scenic alpine driving routes, and outdoor trails.", "category": "Nature"},
    {"name": "Serengeti", "country": "Tanzania", "description": "Vast African savanna famous for annual wildebeest migration, wildlife safaris, and lions.", "category": "Nature"},
    {"name": "Swiss Alps", "country": "Switzerland", "description": "Breathtaking mountain range offering skiing, alpine hiking trails, scenic train rides, and cozy villages.", "category": "Nature"},
    {"name": "Queenstown", "country": "New Zealand", "description": "Adventure capital offering bungee jumping, skydiving, jet boating, skiing, and stunning fjord scenery.", "category": "Adventure"},
    {"name": "Tokyo", "country": "Japan", "description": "Ultra-modern metropolis blending high-tech skyscrapers, anime culture, neon signs, and incredible street food.", "category": "City"},
    {"name": "New York City", "country": "USA", "description": "Bustling mega-city featuring Broadway shows, iconic skyline, Central Park, diverse food, and nightlife.", "category": "City"},
]

def seed_database():
    metadata.create_all(bind=engine)
    
    with engine.begin() as connection:
        count_stmt = select(func.count()).select_from(destinations)
        count = connection.execute(count_stmt).scalar()

        if count > 0:
            print("Database already populated via Core.")
            return

        print("Encoding text and inserting rows via Core INSERT statements...")
        
        insert_records = []
        for item in destinations_data:
            # 1. Encode into a Python float list (384 dimensions)
            embedding_list = model.encode(item["description"]).tolist()

            # 2. Add raw list directly (no pickling!)
            insert_records.append({
                "name": item["name"],
                "country": item["country"],
                "description": item["description"],
                "category": item["category"],
                "embedding": embedding_list
            })

        stmt = insert(destinations)
        connection.execute(stmt, insert_records)
        print(f"Successfully inserted {len(insert_records)} records into PostgreSQL using Core!")

if __name__ == "__main__":
    seed_database()
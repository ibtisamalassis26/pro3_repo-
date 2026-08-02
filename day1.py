import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# STEP 1
destinations = [
    
    {
        "name": "Paris",
        "country": "France",
        "description": "Famous for romantic vibes, iconic art museums, historical landmarks, and world-class culinary experiences.",
        "category": "Culture",
    },
    {
        "name": "Kyoto",
        "country": "Japan",
        "description": "Famed for classical Buddhist temples, gardens, imperial palaces, traditional wooden houses, and cherry blossoms.",
        "category": "Culture",
    },
    {
        "name": "Rome",
        "country": "Italy",
        "description": "Ancient ruins, Colosseum, Vatican City, rich history, cobblestone streets, and amazing pasta.",
        "category": "Culture",
    },
    {
        "name": "Cairo",
        "country": "Egypt",
        "description": "Home to the Giza Pyramids, ancient Sphinx, Nile river cruises, and vibrant bustling bazaars.",
        "category": "Culture",
    },
    
    {
        "name": "Bali",
        "country": "Indonesia",
        "description": "Tropical paradise featuring serene beaches, volcanic mountains, iconic rice terraces, and spiritual retreats.",
        "category": "Beach",
    },
    {
        "name": "Maldives",
        "country": "Maldives",
        "description": "Luxury overwater bungalows, crystal clear turquoise waters, vibrant coral reefs, and romantic sunsets.",
        "category": "Beach",
    },
    {
        "name": "Cancun",
        "country": "Mexico",
        "description": "White sand Caribbean beaches, lively nightlife resorts, and proximity to Mayan archaeological sites.",
        "category": "Beach",
    },
    {
        "name": "Phuket",
        "country": "Thailand",
        "description": "Exotic islands, energetic nightlife, colorful markets, and clear tropical waters perfect for diving.",
        "category": "Beach",
    },
   
    {
        "name": "Yellowstone",
        "country": "USA",
        "description": "Sprawling wilderness with dramatic geysers, hot springs, deep canyons, and abundant wildlife like bison.",
        "category": "Nature",
    },
    {
        "name": "Banff National Park",
        "country": "Canada",
        "description": "Turquoise glacial lakes, majestic Rocky Mountain peaks, scenic alpine driving routes, and outdoor trails.",
        "category": "Nature",
    },
    {
        "name": "Serengeti",
        "country": "Tanzania",
        "description": "Vast African savanna famous for annual wildebeest migration, wildlife safaris, and lions.",
        "category": "Nature",
    },
    {
        "name": "Swiss Alps",
        "country": "Switzerland",
        "description": "Breathtaking mountain range offering skiing, alpine hiking trails, scenic train rides, and cozy villages.",
        "category": "Nature",
    },
    
    {
        "name": "Queenstown",
        "country": "New Zealand",
        "description": "Adventure capital offering bungee jumping, skydiving, jet boating, skiing, and stunning fjord scenery.",
        "category": "Adventure",
    },
    {
        "name": "Tokyo",
        "country": "Japan",
        "description": "Ultra-modern metropolis blending high-tech skyscrapers, anime culture, neon signs, and incredible street food.",
        "category": "City",
    },
    {
        "name": "New York City",
        "country": "USA",
        "description": "Bustling mega-city featuring Broadway shows, iconic skyline, Central Park, diverse food, and nightlife.",
        "category": "City",
    },
]


df = pd.DataFrame(destinations)
print(f"Loaded {len(df)} destinations into DataFrame.")


# STEP 2 & 3
print("\nLoading sentence-transformer model ('all-MiniLM-L6-v2')...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df["embedding"] = df["description"].apply(
    lambda text: model.encode(text).tolist()
)

vector_dim = len(df["embedding"].iloc[0])
print(f"Embeddings successfully generated!")
print(f"Vector Space Dimension: {vector_dim} (Each destination is a 384D point)")

# STEP 4
def find_similar_destinations(query: str, df: pd.DataFrame, top_k: int = 3):
    """Encodes user prompt and ranks destinations by cosine similarity."""
    
    query_vec = model.encode(query).reshape(1, -1)

   
    doc_vectors = np.vstack(df["embedding"].values)

   
    similarities = cosine_similarity(query_vec, doc_vectors)[0]

    
    results = df.copy()
    results["similarity_score"] = similarities
    results = results.sort_values(by="similarity_score", ascending=False)

    return results.head(top_k)



# STEP 5

test_queries = [
    "romantic city with art and fine dining",
    "tropical beach paradise with diving",
    "snowy mountain adventure with skiing",
    "futuristic city with great nightlife",
]

print("\n" + "=" * 60)
print("RUNNING SEMANTIC SEARCH TESTS")
print("=" * 60)

for query in test_queries:
    print(f"\nUser Query: '{query}'")
    top_matches = find_similar_destinations(query, df, top_k=3)

    for rank, (_, row) in enumerate(top_matches.iterrows(), start=1):
        score_percentage = row["similarity_score"] * 100
        print(
            f"  {rank}. {row['name']} ({row['country']}) - Match: {score_percentage:.1f}%"
        )
        print(f"     Category: {row['category']} | Desc: {row['description'][:75]}...")
import random
from typing import Optional, List, Tuple
from sqlalchemy import select, cast
from db import engine
from models import destinations
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector
from models import search_history

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search_destinations(
    query: str, 
    top_k: int = 3, 
    similarity_threshold: float = 0.25,
    category: Optional[str] = None,
    country: Optional[str] = None
) -> Tuple[List[dict], str]:
    """
    Perform a vector similarity search with optional metadata filters (category, country).
    """
    clean_query = query.strip()
    if not clean_query:
        return [], "No query provided."

    # 1. Generate float vector list
    query_vec = model.encode(clean_query).tolist()

    # 2. Setup pgvector distance metric
    vector_col = cast(destinations.c.embedding, Vector(384))
    cosine_dist = vector_col.cosine_distance(query_vec)

    # 3. Base Query
    stmt = select(
        destinations.c.id,
        destinations.c.name,
        destinations.c.country,
        destinations.c.description,
        destinations.c.category,
        (1.0 - cosine_dist).label("similarity_score"),
    )

    # 4. Add Dynamic Filters (Case-insensitive match)
    if category:
        stmt = stmt.where(destinations.c.category.ilike(category.strip()))
    
    if country:
        stmt = stmt.where(destinations.c.country.ilike(country.strip()))

    # 5. Apply Order and Limit
    stmt = stmt.order_by(cosine_dist.asc()).limit(top_k * 2)

    # 6. Execute Query
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    if not rows:
        return [], "No destinations matched your criteria."

    results = [
        {
            "id": row["id"],
            "name": row["name"],
            "country": row["country"],
            "description": row["description"],
            "category": row["category"],
            "score": float(row["similarity_score"]),
        }
        for row in rows
    ]

    top_score = results[0]["score"]
    
    # 7. Fallback check (only if query is very short or poor match)
    if len(clean_query) < 3 or top_score < similarity_threshold:
        fallback_selection = random.sample(results, min(top_k, len(results)))
        for item in fallback_selection:
            item["score"] = 0.0
        return fallback_selection, "fallback"

    return results[:top_k], "success"


def log_search_query(db_conn, query_text: str, category_filter: str = None, query_embedding: list = None, user_id: str = "default_user"):
    """Inserts a new query log into the search_history table."""
    stmt = search_history.insert().values(
        user_id=user_id,
        query_text=query_text,
        category_filter=category_filter,
        query_embedding=query_embedding
    )
    db_conn.execute(stmt)
    db_conn.commit()

def get_user_search_history(db_conn, user_id: str = "default_user", limit: int = 10):
    """Retrieves recent searches for a specific user."""
    stmt = (
        search_history.select()
        .where(search_history.c.user_id == user_id)
        .order_by(search_history.c.created_at.desc())
        .limit(limit)
    )
    results = db_conn.execute(stmt).fetchall()
    
    history = []
    for row in results:
        # Convert SQLAlchemy Row mapping to dictionary format
        mapping = row._mapping
        history.append({
            "id": mapping["id"],
            "query_text": mapping["query_text"],
            "category_filter": mapping["category_filter"],
            "created_at": mapping["created_at"].isoformat() if mapping["created_at"] else None
        })
    return history
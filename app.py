import random
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse  # <--- MUST BE HERE AT THE TOP
from pydantic import BaseModel
from sqlalchemy import select, cast
from db import engine
from models import destinations
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector
# 1. Initialize FastAPI App & Sentence Transformer Model
app = FastAPI(
    title="Travel Destination Vector Search API",
    description="Vector search & recommendation API powered by SentenceTransformers and PostgreSQL pgvector.",
    version="1.0.0",
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 2. Pydantic Schemas for API Responses
class DestinationResponse(BaseModel):
    id: int
    name: str
    country: str
    description: str
    category: str
    score: float
    explanation: str  # XAI field


class SearchAPIResponse(BaseModel):
    query: str
    status: str
    results_count: int
    results: List[DestinationResponse]


class SimilarAPIResponse(BaseModel):
    target_destination: str
    status: str
    results_count: int
    similar_destinations: List[DestinationResponse]


# 3. Explainable AI (XAI) Helper Function
def generate_explanation(query_or_target: str, destination: dict, is_similar_mode: bool = False) -> str:
    """Generates an Explainable AI (XAI) rationale for a recommendation."""
    category = destination["category"]
    description = destination["description"].lower()
    score = destination["score"]
    
    query_words = set(query_or_target.lower().replace(",", "").split())
    
    topics = {
        "romantic": ["romantic", "couples", "honeymoon", "sunsets", "overwater", "serene"],
        "culture": ["culture", "temples", "history", "ancient", "monuments", "spiritual", "heritage"],
        "nature": ["nature", "mountains", "lakes", "hiking", "trails", "alpine", "parks", "scenic"],
        "beach": ["beach", "tropical", "ocean", "coastal", "island", "turquoise", "diving", "sand"],
        "luxury": ["luxury", "resort", "bungalows", "fine"],
        "adventure": ["adventure", "outdoor", "diving", "hiking", "volcano", "explore"]
    }
    
    matched_topics = []
    for topic, keywords in topics.items():
        if any(kw in query_words for kw in keywords) or any(kw in description for kw in keywords):
            if topic.lower() == category.lower() or any(kw in description for kw in keywords):
                matched_topics.append(topic)
    
    matched_topics = list(dict.fromkeys(matched_topics))[:2]
    
    if is_similar_mode:
        reason = f"Shares a similar '{category}' vibe with {query_or_target}"
        if matched_topics:
            reason += f", highlighting shared themes of {' + '.join(matched_topics)}."
        else:
            reason += f" based on overall description similarity."
    else:
        if matched_topics:
            topics_str = " + ".join(matched_topics)
            reason = f"Matches your interest in {topics_str} ({category} destination)."
        else:
            reason = f"High semantic similarity to your query under the '{category}' category."
            
    if score >= 0.35:
        confidence = "Strong Match"
    elif score >= 0.25:
        confidence = "Good Match"
    else:
        confidence = "Moderate Match"
        
    return f"[{confidence}] {reason}"


# 4. Core Helper Vector Search Function
def perform_vector_search(
    query_vec: List[float],
    top_k: int,
    category_filter: Optional[str] = None,
    country_filter: Optional[str] = None,
    exclude_id: Optional[int] = None,
) -> List[dict]:
    """Helper to perform pgvector cosine distance query with optional filters."""
    vector_col = cast(destinations.c.embedding, Vector(384))
    cosine_dist = vector_col.cosine_distance(query_vec)

    stmt = select(
        destinations.c.id,
        destinations.c.name,
        destinations.c.country,
        destinations.c.description,
        destinations.c.category,
        (1.0 - cosine_dist).label("similarity_score"),
    )

    if category_filter:
        stmt = stmt.where(destinations.c.category.ilike(category_filter.strip()))

    if country_filter:
        stmt = stmt.where(destinations.c.country.ilike(country_filter.strip()))

    if exclude_id is not None:
        stmt = stmt.where(destinations.c.id != exclude_id)

    stmt = stmt.order_by(cosine_dist.asc()).limit(top_k * 2)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    return [
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


# 5. API Endpoints

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Travel Vector Search API!",
        "docs": "Visit /docs for Interactive Swagger Documentation",
    }


@app.get("/search", response_model=SearchAPIResponse)
def search_endpoint(
    q: str = Query(..., description="Text query to search for (e.g. 'romantic beach getaway')"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return"),
    category: Optional[str] = Query(None, description="Optional category filter (e.g. Beach, Culture)"),
    country: Optional[str] = Query(None, description="Optional country filter (e.g. Japan, Mexico)"),
    similarity_threshold: float = Query(0.20, description="Minimum similarity threshold before fallback"),
):
    """Search destinations using semantic vector similarity with dynamic metadata filters and XAI reasoning."""
    clean_query = q.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    query_vec = model.encode(clean_query).tolist()
    
    results = perform_vector_search(
        query_vec=query_vec, 
        top_k=top_k, 
        category_filter=category, 
        country_filter=country
    )

    if not results:
        return SearchAPIResponse(query=clean_query, status="no_results", results_count=0, results=[])

    top_score = results[0]["score"]
    
    # Handle Fallback
    if len(clean_query) < 3 or top_score < similarity_threshold:
        fallback = random.sample(results, min(top_k, len(results)))
        for item in fallback:
            item["score"] = 0.0
            item["explanation"] = "[Fallback Match] Query score was low; displaying popular suggestion."
        return SearchAPIResponse(query=clean_query, status="fallback", results_count=len(fallback), results=fallback)

    # Correct sequence: Slice first, then generate explanations!
    final_results = results[:top_k]
    for item in final_results:
        item["explanation"] = generate_explanation(clean_query, item, is_similar_mode=False)

    return SearchAPIResponse(
        query=clean_query, status="success", results_count=len(final_results), results=final_results
    )


@app.get("/similar/{name}", response_model=SimilarAPIResponse)
def similar_destinations_endpoint(
    name: str,
    top_k: int = Query(3, ge=1, le=10, description="Number of similar places to return"),
):
    """Find places similar to a given destination by retrieving its vector and querying nearest neighbors."""
    clean_name = name.strip()

    find_stmt = select(
        destinations.c.id,
        destinations.c.name,
        destinations.c.embedding
    ).where(destinations.c.name.ilike(clean_name))

    with engine.connect() as conn:
        target_row = conn.execute(find_stmt).mappings().first()

    if not target_row:
        raise HTTPException(
            status_code=404, 
            detail=f"Destination '{clean_name}' not found in database."
        )

    target_id = target_row["id"]
    target_vec = target_row["embedding"]

    if isinstance(target_vec, str):
        target_vec = [float(x) for x in target_vec.strip("[]").split(",")]

    similar_results = perform_vector_search(
        query_vec=list(target_vec),
        top_k=top_k,
        exclude_id=target_id
    )

    # Correct sequence: Slice first, then generate explanations!
    final_results = similar_results[:top_k]
    for item in final_results:
        item["explanation"] = generate_explanation(target_row["name"], item, is_similar_mode=True)

    return SimilarAPIResponse(
        target_destination=target_row["name"],
        status="success",
        results_count=len(final_results),
        similar_destinations=final_results
    )


@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>✈️ AI Travel Recommender</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; margin: 0; padding: 30px; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
            h1 { color: #2c3e50; text-align: center; }
            .search-box { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
            input, select, button { padding: 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 15px; }
            input[type="text"] { flex: 2; min-width: 200px; }
            button { background: #3498db; color: white; font-weight: bold; border: none; cursor: pointer; transition: 0.2s; }
            button:hover { background: #2980b9; }
            .card { background: #fafafa; border-left: 5px solid #3498db; padding: 15px; margin-bottom: 15px; border-radius: 4px; }
            .tag { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
            .score { color: #27ae60; font-weight: bold; float: right; }
            .explanation { font-style: italic; color: #555; margin-top: 8px; background: #fff; padding: 8px; border-radius: 4px; border: 1px dashed #cbd5e1; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✈️ AI Travel Recommender</h1>
            <div class="search-box">
                <input type="text" id="query" placeholder="e.g. romantic culture getaway">
                <select id="category">
                    <option value="">All Categories</option>
                    <option value="Beach">Beach</option>
                    <option value="Culture">Culture</option>
                    <option value="Nature">Nature</option>
                </select>
                <button onclick="doSearch()">Search</button>
            </div>
            <div id="results"></div>
        </div>

        <script>
            async function doSearch() {
                const q = document.getElementById('query').value;
                const category = document.getElementById('category').value;
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = "<p>Searching recommendations...</p>";

                let url = `/search?q=${encodeURIComponent(q)}&top_k=3`;
                if (category) url += `&category=${encodeURIComponent(category)}`;

                try {
                    const res = await fetch(url);
                    const data = await res.json();
                    
                    if (data.results_count === 0) {
                        resultsDiv.innerHTML = "<p>No destinations found.</p>";
                        return;
                    }

                    resultsDiv.innerHTML = data.results.map(item => `
                        <div class="card">
                            <span class="score">Match Score: ${(item.score * 100).toFixed(1)}%</span>
                            <h3>${item.name}, ${item.country} <span class="tag">${item.category}</span></h3>
                            <p>${item.description}</p>
                            <div class="explanation">💡 <b>Why recommended:</b> ${item.explanation}</div>
                        </div>
                    `).join('');
                } catch (err) {
                    resultsDiv.innerHTML = "<p style='color:red;'>Error fetching recommendations.</p>";
                }
            }
        </script>
    </body>
    </html>
    """
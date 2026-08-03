import os
import random
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, cast
from pgvector.sqlalchemy import Vector

from db import engine
from models import destinations, search_history

# 1. Load environment variables
load_dotenv()

# 2. Initialize FastAPI App
app = FastAPI(
    title="Travel Destination Vector Search API",
    description="Vector search & recommendation API powered by SentenceTransformers and PostgreSQL pgvector.",
    version="1.0.0",
)

# 3. Load Models & Clients
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

llm_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=LLM_BASE_URL
)


# 4. Pydantic Schemas
class DestinationResponse(BaseModel):
    id: int
    name: str
    country: str
    description: str
    category: str
    score: float
    explanation: str


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


class SearchHistoryItem(BaseModel):
    id: int
    user_id: str
    query_text: str
    category_filter: Optional[str] = None
    created_at: Optional[str] = None


class HistoryAPIResponse(BaseModel):
    user_id: str
    count: int
    history: List[SearchHistoryItem]


class ChatRequest(BaseModel):
    user_message: str
    category_filter: Optional[str] = None


class ChatResponse(BaseModel):
    user_message: str
    assistant_reply: str
    context_destinations: List[str]


# 5. Core Helper Functions
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


def log_search_query(conn, query_text: str, category_filter: Optional[str] = None, query_embedding: Optional[list] = None, user_id: str = "default_user"):
    """Inserts a search record into PostgreSQL search_history table."""
    stmt = search_history.insert().values(
        user_id=user_id,
        query_text=query_text,
        category_filter=category_filter,
        query_embedding=query_embedding
    )
    conn.execute(stmt)
    conn.commit()


def get_user_search_history(conn, user_id: str = "default_user", limit: int = 10):
    """Retrieves past search queries for a specific user."""
    stmt = (
        search_history.select()
        .where(search_history.c.user_id == user_id)
        .order_by(search_history.c.created_at.desc())
        .limit(limit)
    )
    results = conn.execute(stmt).mappings().all()
    
    return [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "query_text": row["query_text"],
            "category_filter": row["category_filter"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }
        for row in results
    ]


# 6. API Endpoints
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
    user_id: str = Query("default_user", description="User identifier for history tracking")
):
    """Search destinations using semantic vector similarity with dynamic metadata filters, XAI reasoning, and search history logging."""
    clean_query = q.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    query_vec = model.encode(clean_query).tolist()
    
    # Save search query into PostgreSQL search_history
    with engine.connect() as conn:
        log_search_query(
            conn=conn,
            query_text=clean_query,
            category_filter=category,
            query_embedding=query_vec,
            user_id=user_id
        )

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

    final_results = similar_results[:top_k]
    for item in final_results:
        item["explanation"] = generate_explanation(target_row["name"], item, is_similar_mode=True)

    return SimilarAPIResponse(
        target_destination=target_row["name"],
        status="success",
        results_count=len(final_results),
        similar_destinations=final_results
    )


@app.get("/history", response_model=HistoryAPIResponse)
def get_history_endpoint(
    user_id: str = Query("default_user", description="User identifier"),
    limit: int = Query(10, ge=1, le=50, description="Number of history items to return")
):
    """Retrieve search history for a user from PostgreSQL."""
    with engine.connect() as conn:
        history = get_user_search_history(conn=conn, user_id=user_id, limit=limit)
        
    return HistoryAPIResponse(
        user_id=user_id,
        count=len(history),
        history=history
    )


@app.post("/chat", response_model=ChatResponse)
def rag_travel_chat(request: ChatRequest):
    """Retrieves pgvector results and passes them to an LLM to provide a conversational answer."""
    clean_message = request.user_message.strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    query_vec = model.encode(clean_message).tolist()
    raw_results = perform_vector_search(
        query_vec=query_vec,
        top_k=3,
        category_filter=request.category_filter
    )
    
    if not raw_results:
        return ChatResponse(
            user_message=clean_message,
            assistant_reply="I couldn't find any destinations matching your criteria in our database.",
            context_destinations=[]
        )

    context_text = ""
    destination_names = []
    for dest in raw_results:
        destination_names.append(dest["name"])
        context_text += f"- **{dest['name']}, {dest['country']}** ({dest['category']}): {dest['description']}\n"

    system_prompt = (
        "You are an expert, friendly AI travel advisor. "
        "Answer the user's question using ONLY the destination information provided below. "
        "Keep your response warm, helpful, and concise.\n\n"
        f"DESTINATION CONTEXT:\n{context_text}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_message}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"I retrieved these top destinations ({', '.join(destination_names)}), but couldn't reach the LLM generator. Error: {str(e)}"

    return ChatResponse(
        user_message=clean_message,
        assistant_reply=reply,
        context_destinations=destination_names
    )


@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>✈️ AI Travel Recommender & Assistant</title>
        <style>
            :root {
                --primary: #2563eb;
                --primary-dark: #1d4ed8;
                --llm-color: #7c3aed;
                --llm-bg: #f5f3ff;
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
                --border-color: #e2e8f0;
            }

            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                background: var(--bg-color); 
                color: var(--text-main);
                margin: 0; 
                padding: 40px 20px; 
            }

            .container { 
                max-width: 850px; 
                margin: 0 auto; 
                background: var(--card-bg); 
                padding: 35px; 
                border-radius: 16px; 
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05); 
            }

            h1 { 
                color: var(--text-main); 
                text-align: center; 
                margin-top: 0;
                margin-bottom: 25px;
                font-size: 28px;
            }

            .search-box { 
                display: flex; 
                gap: 12px; 
                margin-bottom: 30px; 
                flex-wrap: wrap; 
            }

            input, select, button { 
                padding: 14px 16px; 
                border: 1px solid var(--border-color); 
                border-radius: 8px; 
                font-size: 15px; 
                outline: none;
                transition: all 0.2s ease;
            }

            input[type="text"] { 
                flex: 2; 
                min-width: 220px; 
            }

            input[type="text"]:focus, select:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }

            button { 
                background: var(--primary); 
                color: white; 
                font-weight: 600; 
                border: none; 
                cursor: pointer; 
            }

            button:hover { 
                background: var(--primary-dark); 
            }

            .btn-llm { 
                background: var(--llm-color); 
            }
            .btn-llm:hover { 
                background: #6d28d9; 
            }

            .btn-similar { 
                background: #f97316; 
                padding: 8px 14px; 
                font-size: 13px; 
                margin-top: 12px; 
                border-radius: 6px;
            }
            .btn-similar:hover { 
                background: #ea580c; 
            }

            .card { 
                background: #ffffff; 
                border: 1px solid var(--border-color);
                border-left: 6px solid var(--primary); 
                padding: 20px; 
                margin-bottom: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }

            .card h3 {
                margin: 0 0 8px 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .chat-response { 
                background: var(--llm-bg); 
                border: 1px solid #ddd6fe;
                border-left: 6px solid var(--llm-color); 
                padding: 24px; 
                border-radius: 12px; 
                margin-top: 20px; 
                line-height: 1.7; 
            }

            .chat-response h3 {
                margin-top: 0;
                color: #5b21b6;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .tag { 
                display: inline-block; 
                background: #eff6ff; 
                color: var(--primary); 
                padding: 4px 10px; 
                border-radius: 20px; 
                font-size: 12px; 
                font-weight: 600; 
            }

            .score { 
                color: #16a34a; 
                font-size: 14px;
                font-weight: 600; 
                background: #f0fdf4;
                padding: 4px 10px;
                border-radius: 20px;
            }

            .explanation { 
                font-style: italic; 
                color: var(--text-muted); 
                margin-top: 12px; 
                background: #f8fafc; 
                padding: 12px; 
                border-radius: 6px; 
                border: 1px dashed var(--border-color); 
                font-size: 14px;
            }

            .loading {
                text-align: center;
                color: var(--text-muted);
                padding: 20px;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✈️ AI Travel Recommender & Assistant</h1>
            
            <div class="search-box">
                <input type="text" id="query" placeholder="e.g. romantic beach getaway">
                <select id="category">
                    <option value="">All Categories</option>
                    <option value="Beach">Beach</option>
                    <option value="Culture">Culture</option>
                    <option value="Nature">Nature</option>
                </select>
                <button onclick="doSearch()">Search</button>
                <button class="btn-llm" onclick="doChat()">Ask LLM Advisor 🤖</button>
            </div>

            <div id="results"></div>
        </div>

        <script>
            // Clean up text formatting helper (removes markdown asterisks if any slip through)
            function cleanMarkdown(text) {
                if (!text) return '';
                return text.replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>').replace(/\\*(.*?)\\*/g, '<i>$1</i>');
            }

            // Standard Vector Search
            async function doSearch() {
                const q = document.getElementById('query').value;
                const category = document.getElementById('category').value;
                const resultsDiv = document.getElementById('results');
                
                if (!q) { alert("Please enter a query!"); return; }
                resultsDiv.innerHTML = "<div class='loading'>Searching vector database...</div>";

                let url = `/search?q=${encodeURIComponent(q)}&top_k=3`;
                if (category) url += `&category=${encodeURIComponent(category)}`;

                try {
                    const res = await fetch(url);
                    const data = await res.json();
                    
                    if (data.results_count === 0) {
                        resultsDiv.innerHTML = "<p class='loading'>No destinations found matching your criteria.</p>";
                        return;
                    }

                    resultsDiv.innerHTML = data.results.map(item => `
                        <div class="card">
                            <h3>
                                <span>${item.name}, ${item.country} <span class="tag">${item.category}</span></span>
                                <span class="score">Match: ${(item.score * 100).toFixed(1)}%</span>
                            </h3>
                            <p>${item.description}</p>
                            <div class="explanation">💡 <b>Why recommended:</b> ${item.explanation}</div>
                            <button class="btn-similar" onclick="findSimilar('${item.name}')">Find Places Similar to ${item.name} 🔗</button>
                        </div>
                    `).join('');
                } catch (err) {
                    resultsDiv.innerHTML = "<p style='color:red; text-align:center;'>Error fetching recommendations.</p>";
                }
            }

            // LLM RAG Chat Assistant
            async function doChat() {
                const q = document.getElementById('query').value;
                const category = document.getElementById('category').value;
                const resultsDiv = document.getElementById('results');
                
                if (!q) { alert("Please enter a question or request for the LLM!"); return; }
                resultsDiv.innerHTML = "<div class='loading'>🤖 Consulting AI LLM Advisor with context...</div>";

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_message: q, category_filter: category || null })
                    });
                    
                    const data = await res.json();
                    
                    // Format response neatly
                    let formattedReply = data.assistant_reply
                        .replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>')
                        .replace(/\\n/g, '<br>');

                    resultsDiv.innerHTML = `
                        <div class="chat-response">
                            <h3>🤖 AI Travel Advisor Response</h3>
                            <p>${formattedReply}</p>
                            <hr style="border:0; border-top: 1px solid #ddd6fe; margin: 15px 0;">
                            <small style="color: #6d28d9;"><b>Context Retrieved from pgvector:</b> ${data.context_destinations.join(', ')}</small>
                        </div>
                    `;
                } catch (err) {
                    resultsDiv.innerHTML = "<p style='color:red; text-align:center;'>Error connecting to LLM Assistant.</p>";
                }
            }

            // Find Similar Destinations Endpoint
            async function findSimilar(destinationName) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `<div class='loading'>Finding places similar to <b>${destinationName}</b>...</div>`;

                try {
                    const res = await fetch(`/similar/${encodeURIComponent(destinationName)}?top_k=3`);
                    const data = await res.json();

                    resultsDiv.innerHTML = `<h2 style="font-size: 20px; color: #334155; margin-bottom: 15px;">Places Similar to "${data.target_destination}"</h2>` + 
                    data.similar_destinations.map(item => `
                        <div class="card" style="border-left-color: #f97316;">
                            <h3>
                                <span>${item.name}, ${item.country} <span class="tag">${item.category}</span></span>
                                <span class="score">Similarity: ${(item.score * 100).toFixed(1)}%</span>
                            </h3>
                            <p>${item.description}</p>
                            <div class="explanation">💡 <b>Why similar:</b> ${item.explanation}</div>
                        </div>
                    `).join('');
                } catch (err) {
                    resultsDiv.innerHTML = "<p style='color:red; text-align:center;'>Error fetching similar destinations.</p>";
                }
            }
        </script>
    </body>
    </html>
    """
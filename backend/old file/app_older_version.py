from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys

# Add the services directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

# Import our services
from simple_pubmed_client import SimplePubMedClient
from rag_system import RAGSystem
from vector_search_system import VectorSearchSystem

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Initialize services (same as before)
services_status = {}

try:
    pubmed_client = SimplePubMedClient()
    services_status['pubmed'] = True
    print("✅ PubMed client initialized successfully")
except Exception as e:
    services_status['pubmed'] = False
    pubmed_client = None

try:
    rag_system = RAGSystem()
    services_status['rag'] = True
    print("✅ RAG system initialized successfully")
except Exception as e:
    services_status['rag'] = False
    rag_system = None

try:
    vector_system = VectorSearchSystem()
    services_status['vector_search'] = True
    print("✅ Vector search system initialized successfully")
except Exception as e:
    services_status['vector_search'] = False
    vector_system = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced PubMed RAG API with Improved Search",
        "version": "3.1.0",
        "services": services_status
    })

def smart_search_strategy(query: str, max_results: int = 5):
    """
    Improved search strategy that prioritizes fresh PubMed results
    and uses vector search more intelligently
    """
    print(f"🧠 Using smart search strategy for: '{query}'")
    
    # Step 1: Always get fresh PubMed results first
    pubmed_articles = pubmed_client.search_and_get_articles(query, max_results=max_results)
    print(f"📚 PubMed found: {len(pubmed_articles)} articles")
    
    if not pubmed_articles:
        print("⚠️ No PubMed articles found")
        return [], {"strategy": "no_results", "pubmed_count": 0, "vector_count": 0}
    
    # Step 2: Check if we have enough high-quality PubMed results
    quality_pubmed = [
        article for article in pubmed_articles 
        if article.get('abstract', '').strip() and 
           article.get('abstract') != 'Abstract not available' and
           len(article.get('abstract', '')) > 100
    ]
    
    print(f"📊 High-quality PubMed articles: {len(quality_pubmed)}")
    
    # Step 3: Decide search strategy based on PubMed results quality
    if len(quality_pubmed) >= 3:
        # We have enough good PubMed results - prioritize them
        print("✅ Sufficient high-quality PubMed results - using PubMed-priority strategy")
        
        # Add to vector DB for future searches but return PubMed results
        if vector_system:
            vector_system.add_articles_to_vector_db(pubmed_articles)
        
        return pubmed_articles, {
            "strategy": "pubmed_priority", 
            "pubmed_count": len(pubmed_articles),
            "quality_count": len(quality_pubmed),
            "reasoning": f"Found {len(quality_pubmed)} high-quality articles from PubMed"
        }
    
    elif len(quality_pubmed) >= 1 and vector_system:
        # Some PubMed results but could benefit from vector search
        print("🔄 Moderate PubMed results - using hybrid enhancement")
        
        # Add PubMed articles to vector DB
        vector_system.add_articles_to_vector_db(pubmed_articles)
        
        # Get additional semantically similar articles, but filter smartly
        vector_articles = vector_system.semantic_search(query, n_results=max_results * 2)
        
        # Filter vector articles to only include highly relevant ones
        relevant_vector = [
            article for article in vector_articles 
            if article.get('similarity_score', 0) > 0.75  # Only very similar articles
        ]
        
        # Combine: prioritize fresh PubMed, then add highly relevant vector results
        seen_pmids = {article['pmid'] for article in pubmed_articles}
        for v_article in relevant_vector:
            if len(pubmed_articles) >= max_results:
                break
            if v_article['pmid'] not in seen_pmids:
                pubmed_articles.append(v_article)
                seen_pmids.add(v_article['pmid'])
        
        return pubmed_articles[:max_results], {
            "strategy": "hybrid_enhanced",
            "pubmed_count": len(quality_pubmed),
            "vector_added": len(relevant_vector),
            "reasoning": f"Enhanced {len(quality_pubmed)} PubMed articles with {len(relevant_vector)} highly relevant vector results"
        }
    
    else:
        # Poor PubMed results - rely more on vector search if available
        print("⚠️ Limited PubMed results - using vector-enhanced strategy")
        
        if vector_system and vector_system.collection.count() > 0:
            # Add new articles and search semantically
            vector_system.add_articles_to_vector_db(pubmed_articles)
            vector_articles = vector_system.semantic_search(query, n_results=max_results)
            
            # Combine with bias toward fresh PubMed results
            combined = []
            seen_pmids = set()
            
            # First add PubMed results (fresh and keyword-matched)
            for article in pubmed_articles:
                if article['pmid'] not in seen_pmids:
                    combined.append(article)
                    seen_pmids.add(article['pmid'])
            
            # Then add vector results if similarity is decent
            for v_article in vector_articles:
                if len(combined) >= max_results:
                    break
                if (v_article['pmid'] not in seen_pmids and 
                    v_article.get('similarity_score', 0) > 0.6):
                    combined.append(v_article)
                    seen_pmids.add(v_article['pmid'])
            
            return combined[:max_results], {
                "strategy": "vector_enhanced",
                "pubmed_count": len(pubmed_articles),
                "vector_count": len([a for a in vector_articles if a.get('similarity_score', 0) > 0.6]),
                "reasoning": f"Limited PubMed results enhanced with vector search"
            }
        else:
            # No vector search available or empty DB
            return pubmed_articles, {
                "strategy": "pubmed_only",
                "pubmed_count": len(pubmed_articles),
                "reasoning": "Vector search unavailable, using PubMed results only"
            }

@app.route('/search', methods=['POST'])
def search_pubmed():
    """Enhanced search with improved strategy"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        max_results = data.get('max_results', 5)
        
        print(f"🔍 Received search query: '{query}'")
        
        if not pubmed_client:
            return jsonify({
                "error": "PubMed service unavailable",
                "message": "PubMed client could not be initialized."
            }), 503
        
        # Use improved search strategy
        articles, search_metadata = smart_search_strategy(query, max_results)
        
        if not articles:
            return jsonify({
                "query": query,
                "answer": "No relevant articles found for your query. Please try different search terms.",
                "articles": [],
                "status": "no_results",
                "search_metadata": search_metadata
            })
        
        # Generate AI answer
        if rag_system:
            print(f"🤖 Generating AI answer using {len(articles)} articles...")
            rag_result = rag_system.generate_answer(query, articles)
            
            response = {
                "query": query,
                "answer": rag_result["answer"],
                "articles": articles,
                "status": "success",
                "total_found": len(articles),
                "rag_enabled": True,
                "search_metadata": search_metadata,
                "ai_metadata": {
                    "confidence": rag_result["confidence"],
                    "sources_used": rag_result["sources_used"],
                    "reasoning": rag_result["reasoning"],
                    "tokens_used": rag_result.get("tokens_used", 0)
                }
            }
            
        else:
            # Fallback without RAG
            answer = f"Found {len(articles)} relevant articles about '{query}' using {search_metadata['strategy']} search strategy."
            
            response = {
                "query": query,
                "answer": answer,
                "articles": articles,
                "status": "success",
                "total_found": len(articles),
                "rag_enabled": False,
                "search_metadata": search_metadata
            }
        
        print(f"✅ Returning {len(articles)} articles using {search_metadata['strategy']} strategy")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in search endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

# Keep other endpoints the same
@app.route('/ask', methods=['POST'])
def ask_followup():
    """Follow-up questions endpoint"""
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'articles' not in data:
            return jsonify({"error": "Query and articles context are required"}), 400
        
        query = data['query'].strip()
        articles = data['articles']
        
        if not query or not rag_system:
            return jsonify({"error": "Invalid request or RAG system unavailable"}), 400
        
        print(f"❓ Follow-up question: '{query}'")
        rag_result = rag_system.generate_answer(query, articles)
        
        return jsonify({
            "query": query,
            "answer": rag_result["answer"],
            "articles": articles,
            "status": "success",
            "is_followup": True,
            "ai_metadata": {
                "confidence": rag_result["confidence"],
                "sources_used": rag_result["sources_used"],
                "reasoning": rag_result["reasoning"],
                "tokens_used": rag_result.get("tokens_used", 0)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting Improved PubMed RAG API...")
    print(f"🔍 Server: http://{host}:{port}")
    print(f"🧠 Smart search strategies enabled")
    
    app.run(host=host, port=port, debug=debug)
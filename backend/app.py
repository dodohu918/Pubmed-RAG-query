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

# Configuration - Set debug=False for production
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Initialize services with caching
services_status = {}

# Initialize PubMed client
try:
    pubmed_client = SimplePubMedClient()
    services_status['pubmed'] = True
    print("✅ PubMed client initialized successfully")
except Exception as e:
    services_status['pubmed'] = False
    pubmed_client = None
    print(f"⚠️ Warning: PubMed client initialization failed: {e}")

# Initialize RAG system
try:
    rag_system = RAGSystem()
    services_status['rag'] = True
    print("✅ RAG system initialized successfully")
except Exception as e:
    services_status['rag'] = False
    rag_system = None
    print(f"⚠️ Warning: RAG system initialization failed: {e}")

# Initialize Vector Search system
try:
    vector_system = VectorSearchSystem()
    services_status['vector_search'] = True
    print("✅ Vector search system initialized successfully")
except Exception as e:
    services_status['vector_search'] = False
    vector_system = None
    print(f"⚠️ Warning: Vector search initialization failed: {e}")

@app.route('/', methods=['GET'])
def health_check():
    """Enhanced health check with service status"""
    return jsonify({
        "status": "healthy",
        "message": "PubMed RAG API - Live on Railway! 🚀",
        "version": "4.0.0",
        "services": services_status,
        "features": {
            "pubmed_search": services_status.get('pubmed', False),
            "ai_answers": services_status.get('rag', False),
            "semantic_search": services_status.get('vector_search', False),
            "hybrid_search": services_status.get('pubmed', False) and services_status.get('vector_search', False)
        }
    })

def smart_search_strategy_cached(query: str, max_results: int = 5):
    """
    Smart search strategy optimized for production
    """
    print(f"🔍 Smart search for: '{query}'")
    
    # Step 1: Get PubMed results
    pubmed_articles = pubmed_client.search_and_get_articles(query, max_results)
    print(f"📚 PubMed found: {len(pubmed_articles)} articles")
    
    if not pubmed_articles:
        return [], {"strategy": "no_results", "pubmed_count": 0, "vector_count": 0}
    
    # Step 2: Check quality
    quality_pubmed = [
        article for article in pubmed_articles 
        if article.get('abstract', '').strip() and 
           article.get('abstract') != 'Abstract not available' and
           len(article.get('abstract', '')) > 100
    ]
    
    print(f"📊 High-quality articles: {len(quality_pubmed)}")
    
    # Step 3: Apply strategy
    if len(quality_pubmed) >= 3:
        # Sufficient quality - use PubMed priority
        if vector_system:
            vector_system.add_articles_to_vector_db(pubmed_articles)
        
        return pubmed_articles, {
            "strategy": "pubmed_priority", 
            "pubmed_count": len(pubmed_articles),
            "quality_count": len(quality_pubmed),
            "reasoning": f"Found {len(quality_pubmed)} high-quality articles"
        }
    
    elif len(quality_pubmed) >= 1 and vector_system:
        # Hybrid enhancement
        vector_system.add_articles_to_vector_db(pubmed_articles)
        vector_articles = vector_system.semantic_search(query, n_results=max_results * 2)
        
        relevant_vector = [
            article for article in vector_articles 
            if article.get('similarity_score', 0) > 0.75
        ]
        
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
            "reasoning": f"Enhanced with {len(relevant_vector)} relevant vector results"
        }
    
    else:
        # Fallback to PubMed only
        return pubmed_articles, {
            "strategy": "pubmed_only",
            "pubmed_count": len(pubmed_articles),
            "reasoning": "Using PubMed results"
        }

@app.route('/search', methods=['POST'])
def search_pubmed():
    """Main search endpoint"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        max_results = data.get('max_results', 5)
        
        print(f"🔍 Search query: '{query}'")
        
        if not pubmed_client:
            return jsonify({
                "error": "PubMed service unavailable",
                "message": "PubMed client could not be initialized."
            }), 503
        
        # Perform smart search
        articles, search_metadata = smart_search_strategy_cached(query, max_results)
        
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
            print(f"🤖 Generating AI answer...")
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
            response = {
                "query": query,
                "answer": f"Found {len(articles)} relevant articles about '{query}'.",
                "articles": articles,
                "status": "success",
                "total_found": len(articles),
                "rag_enabled": False,
                "search_metadata": search_metadata
            }
        
        print(f"✅ Returning {len(articles)} articles")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

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
        
        print(f"❓ Follow-up: '{query}'")
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Production configuration for Railway
    port = int(os.environ.get('PORT', 5000))  # Railway provides PORT
    host = '0.0.0.0'  # Allow external connections
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting PubMed RAG API on Railway...")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Services initialized:")
    for service, status in services_status.items():
        print(f"   {service}: {'✅' if status else '❌'}")
    
    app.run(host=host, port=port, debug=debug)
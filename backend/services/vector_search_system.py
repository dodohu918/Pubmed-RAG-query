import numpy as np
from openai import OpenAI
import chromadb
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import hashlib
import json

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class VectorSearchSystem:
    """
    Vector-based semantic search system for PubMed articles using embeddings
    """
    
    def __init__(self, api_key: str = None):
        """Initialize vector search system"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required for embeddings.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize ChromaDB (persistent local database)
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        
        # Create or get collection for PubMed articles
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_articles",
            metadata={"description": "PubMed article embeddings for semantic search"}
        )
        
        print("✅ Vector search system initialized")
        print(f"📊 Current collection size: {self.collection.count()} articles")
    
    def add_articles_to_vector_db(self, articles: List[Dict]) -> int:
        """
        Add articles to vector database with embeddings
        
        Args:
            articles: List of PubMed articles with abstracts
            
        Returns:
            Number of articles actually added (skips duplicates)
        """
        if not articles:
            return 0
        
        new_articles = []
        texts_to_embed = []
        
        print(f"🔄 Processing {len(articles)} articles for vector database...")
        
        for article in articles:
            pmid = article.get('pmid', '')
            if not pmid:
                continue
            
            # Check if article already exists
            try:
                existing = self.collection.get(ids=[pmid])
                if existing['ids']:
                    print(f"⏭️  Skipping PMID {pmid} (already in database)")
                    continue
            except:
                pass  # Article doesn't exist, we'll add it
            
            # Prepare text for embedding (title + abstract)
            title = article.get('title', '').strip()
            abstract = article.get('abstract', '').strip()
            
            if not abstract or abstract == 'Abstract not available' or len(abstract) < 50:
                print(f"⏭️  Skipping PMID {pmid} (no meaningful abstract)")
                continue
            
            # Combine title and abstract for better embedding
            combined_text = f"Title: {title}\n\nAbstract: {abstract}"
            
            new_articles.append({
                'id': pmid,
                'text': combined_text,
                'metadata': {
                    'title': title,
                    'authors': json.dumps(article.get('authors', [])),
                    'journal': article.get('journal', ''),
                    'pub_date': article.get('pub_date', ''),
                    'url': article.get('url', ''),
                    'abstract': abstract
                }
            })
            texts_to_embed.append(combined_text)
        
        if not new_articles:
            print("📊 No new articles to add to vector database")
            return 0
        
        # Generate embeddings for all new articles
        print(f"🧠 Generating embeddings for {len(texts_to_embed)} articles...")
        embeddings = self._get_embeddings(texts_to_embed)
        
        if not embeddings:
            print("❌ Failed to generate embeddings")
            return 0
        
        # Add to ChromaDB
        ids = [article['id'] for article in new_articles]
        documents = [article['text'] for article in new_articles]
        metadatas = [article['metadata'] for article in new_articles]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(new_articles)} articles to vector database")
        return len(new_articles)
    
    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Perform semantic search using embeddings
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            
        Returns:
            List of most relevant articles with similarity scores
        """
        if self.collection.count() == 0:
            print("⚠️  Vector database is empty. Add articles first.")
            return []
        
        print(f"🔍 Performing semantic search for: '{query}'")
        
        # Generate embedding for the query
        query_embedding = self._get_embeddings([query])
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count())
        )
        
        # Convert results to article format
        articles = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
            
            article = {
                'pmid': results['ids'][0][i],
                'title': metadata.get('title', ''),
                'authors': json.loads(metadata.get('authors', '[]')),
                'journal': metadata.get('journal', ''),
                'pub_date': metadata.get('pub_date', ''),
                'abstract': metadata.get('abstract', ''),
                'url': metadata.get('url', ''),
                'similarity_score': round(similarity, 3),
                'has_abstract': True
            }
            articles.append(article)
        
        print(f"🎯 Found {len(articles)} semantically similar articles")
        return articles
    
    def hybrid_search(self, query: str, pubmed_articles: List[Dict], n_results: int = 10) -> List[Dict]:
        """
        Combine PubMed keyword search with semantic vector search
        
        Args:
            query: Search query
            pubmed_articles: Articles from PubMed keyword search
            n_results: Maximum number of results to return
            
        Returns:
            Ranked and deduplicated articles from both sources
        """
        print(f"🔄 Performing hybrid search for: '{query}'")
        
        # Add new PubMed articles to vector database
        added_count = self.add_articles_to_vector_db(pubmed_articles)
        if added_count > 0:
            print(f"📊 Added {added_count} new articles to vector database")
        
        # Get semantic search results
        vector_articles = self.semantic_search(query, n_results * 2)  # Get more for better ranking
        
        # Combine and rank results
        combined_articles = self._combine_and_rank_results(
            pubmed_articles, vector_articles, query, n_results
        )
        
        print(f"✅ Hybrid search returned {len(combined_articles)} articles")
        return combined_articles
    
    def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings using OpenAI's text-embedding model"""
        try:
            # Use OpenAI's latest embedding model
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Faster and cheaper than ada-002
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            print(f"🧠 Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            return None
    
    def _combine_and_rank_results(self, pubmed_articles: List[Dict], 
                                vector_articles: List[Dict], 
                                query: str, n_results: int) -> List[Dict]:
        """Combine and rank results from both search methods"""
        
        # Create a dictionary to avoid duplicates
        article_dict = {}
        
        # Add PubMed articles (give them keyword match bonus)
        for i, article in enumerate(pubmed_articles[:n_results]):
            pmid = article.get('pmid', '')
            if pmid:
                article['rank_score'] = 1.0 - (i * 0.1)  # Higher score for higher PubMed rank
                article['search_source'] = 'pubmed'
                article_dict[pmid] = article
        
        # Add vector search articles (they already have similarity scores)
        for i, article in enumerate(vector_articles[:n_results]):
            pmid = article.get('pmid', '')
            if pmid:
                if pmid in article_dict:
                    # Article found in both searches - boost its score
                    article_dict[pmid]['rank_score'] = (
                        article_dict[pmid].get('rank_score', 0) + 
                        article['similarity_score'] + 0.3  # Bonus for appearing in both
                    )
                    article_dict[pmid]['search_source'] = 'both'
                    article_dict[pmid]['similarity_score'] = article['similarity_score']
                else:
                    # Only in vector search
                    article['rank_score'] = article['similarity_score']
                    article['search_source'] = 'vector'
                    article_dict[pmid] = article
        
        # Sort by combined rank score
        ranked_articles = sorted(
            article_dict.values(), 
            key=lambda x: x.get('rank_score', 0), 
            reverse=True
        )
        
        return ranked_articles[:n_results]
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database"""
        count = self.collection.count()
        return {
            "total_articles": count,
            "database_path": "./vector_db",
            "embedding_model": "text-embedding-3-small",
            "collection_name": "pubmed_articles"
        }
    
    def clear_database(self):
        """Clear the vector database (use with caution!)"""
        self.chroma_client.delete_collection("pubmed_articles")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pubmed_articles",
            metadata={"description": "PubMed article embeddings for semantic search"}
        )
        print("🗑️  Vector database cleared")

def test_vector_search():
    """Test the vector search system"""
    try:
        # Test with sample data
        vector_system = VectorSearchSystem()
        
        sample_articles = [
            {
                'pmid': '12345',
                'title': 'Exercise Training and Cardiovascular Health in Type 2 Diabetes',
                'authors': ['Smith J', 'Johnson A'],
                'journal': 'Diabetes Care',
                'pub_date': '2023',
                'abstract': 'Regular aerobic exercise training significantly improves cardiovascular outcomes in patients with type 2 diabetes mellitus. This study demonstrates reduced HbA1c, improved insulin sensitivity, and decreased cardiovascular risk markers.',
                'url': 'https://pubmed.ncbi.nlm.nih.gov/12345/'
            },
            {
                'pmid': '12346',
                'title': 'Mediterranean Diet and Heart Disease Prevention',
                'authors': ['Davis M', 'Wilson R'],
                'journal': 'New England Journal of Medicine',
                'pub_date': '2023',
                'abstract': 'The Mediterranean dietary pattern shows significant protective effects against coronary heart disease and myocardial infarction. Olive oil, nuts, and fish consumption correlate with reduced cardiovascular mortality.',
                'url': 'https://pubmed.ncbi.nlm.nih.gov/12346/'
            }
        ]
        
        # Add articles to vector database
        added = vector_system.add_articles_to_vector_db(sample_articles)
        print(f"✅ Added {added} articles to test database")
        
        # Test semantic search
        results = vector_system.semantic_search("heart attack prevention", n_results=5)
        
        print(f"\n🧪 Vector Search Test Results:")
        for i, article in enumerate(results, 1):
            print(f"{i}. {article['title']}")
            print(f"   Similarity: {article['similarity_score']}")
            print(f"   PMID: {article['pmid']}")
        
        # Get stats
        stats = vector_system.get_database_stats()
        print(f"\n📊 Database Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_vector_search()
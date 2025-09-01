from openai import OpenAI
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class RAGSystem:
    """
    Retrieval-Augmented Generation system that uses PubMed articles to generate
    intelligent answers to medical/scientific questions.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize RAG system with OpenAI API"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in your .env file.")
        
        # Initialize OpenAI client (new v1.0+ format)
        self.client = OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = "gpt-4"  # Use GPT-4 for better medical/scientific responses
        self.max_tokens = 1000
        self.temperature = 0.3  # Lower temperature for more factual responses
        
    def generate_answer(self, query: str, articles: List[Dict]) -> Dict:
        """
        Generate an intelligent answer using the query and retrieved articles
        
        Args:
            query: User's search query
            articles: List of PubMed articles with abstracts
            
        Returns:
            Dictionary with generated answer, confidence, and metadata
        """
        try:
            # Prepare context from articles
            context = self._prepare_context(articles)
            
            if not context.strip():
                return {
                    "answer": "I couldn't find sufficient information in the retrieved articles to answer your question.",
                    "confidence": "low",
                    "sources_used": 0,
                    "reasoning": "No usable abstracts found in the retrieved articles."
                }
            
            # Create the prompt
            prompt = self._create_prompt(query, context, articles)
            
            print(f"🤖 Generating AI answer for: '{query}'")
            print(f"📚 Using {len(articles)} articles as context")
            
            # Generate response using OpenAI (new v1.0+ format)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical and scientific research assistant. Your role is to provide accurate, evidence-based answers using only the scientific literature provided. 

IMPORTANT GUIDELINES:
- Only use information from the provided abstracts
- Be precise and factual
- Cite specific studies when making claims
- If information is insufficient, say so honestly
- Use medical terminology appropriately but explain complex terms
- Focus on evidence-based conclusions
- Never make treatment recommendations - only describe what research shows"""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Analyze the response quality
            analysis = self._analyze_response_quality(answer, articles)
            
            return {
                "answer": answer,
                "confidence": analysis["confidence"],
                "sources_used": analysis["sources_used"],
                "reasoning": analysis["reasoning"],
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            print(f"❌ Error generating RAG answer: {str(e)}")
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "confidence": "error",
                "sources_used": 0,
                "reasoning": f"Error in RAG generation: {str(e)}"
            }
    
    def _prepare_context(self, articles: List[Dict]) -> str:
        """Prepare context string from articles"""
        context_parts = []
        
        for i, article in enumerate(articles, 1):
            # Only include articles with meaningful abstracts
            abstract = article.get('abstract', '').strip()
            if abstract and abstract != 'Abstract not available' and len(abstract) > 50:
                context_part = f"""
Study {i}:
Title: {article.get('title', 'Unknown title')}
Authors: {', '.join(article.get('authors', ['Unknown authors'])[:3])}
Journal: {article.get('journal', 'Unknown journal')} ({article.get('pub_date', 'Unknown date')})
PMID: {article.get('pmid', 'Unknown')}
Abstract: {abstract}
---"""
                context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, articles: List[Dict]) -> str:
        """Create the prompt for OpenAI"""
        return f"""Based on the following scientific literature, please answer this question: "{query}"

SCIENTIFIC CONTEXT:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Synthesizes information from the provided studies
3. Mentions specific findings when relevant
4. Notes any limitations or conflicting results
5. Maintains scientific accuracy

If the provided literature doesn't contain sufficient information to answer the question, please state this clearly.

Answer:"""
    
    def _analyze_response_quality(self, answer: str, articles: List[Dict]) -> Dict:
        """Analyze the quality and confidence of the generated response"""
        
        # Count articles with usable abstracts
        usable_articles = sum(1 for article in articles 
                            if article.get('abstract', '').strip() 
                            and article.get('abstract') != 'Abstract not available' 
                            and len(article.get('abstract', '')) > 50)
        
        # Simple heuristics for confidence
        answer_length = len(answer.split())
        
        if "insufficient information" in answer.lower() or "cannot answer" in answer.lower():
            confidence = "low"
            reasoning = "AI indicated insufficient information to answer the question"
        elif usable_articles >= 3 and answer_length >= 100:
            confidence = "high"
            reasoning = f"Strong evidence base with {usable_articles} relevant studies"
        elif usable_articles >= 2 and answer_length >= 50:
            confidence = "medium"
            reasoning = f"Moderate evidence base with {usable_articles} relevant studies"
        else:
            confidence = "low"
            reasoning = f"Limited evidence base with only {usable_articles} relevant studies"
        
        return {
            "confidence": confidence,
            "sources_used": usable_articles,
            "reasoning": reasoning
        }

def test_rag_system():
    """Test the RAG system with sample data"""
    try:
        rag = RAGSystem()
        
        # Sample articles data (would come from PubMed)
        sample_articles = [
            {
                'pmid': '12345',
                'title': 'Effects of Exercise on Type 2 Diabetes',
                'authors': ['Smith J', 'Johnson A', 'Brown K'],
                'journal': 'Diabetes Care',
                'pub_date': '2023',
                'abstract': 'This randomized controlled trial examined the effects of regular exercise on glycemic control in 200 patients with type 2 diabetes. Participants who engaged in 150 minutes of moderate-intensity exercise per week showed significant improvements in HbA1c levels (mean reduction of 0.7%) compared to the control group. The exercise intervention also led to improvements in insulin sensitivity and cardiovascular risk factors.'
            },
            {
                'pmid': '12346',
                'title': 'Dietary Interventions in Diabetes Management',
                'authors': ['Davis M', 'Wilson R'],
                'journal': 'Journal of Nutrition',
                'pub_date': '2023',
                'abstract': 'A systematic review of 25 studies examining dietary interventions for type 2 diabetes management. Low-carbohydrate diets showed the most consistent improvements in glycemic control, with average HbA1c reductions of 0.5-1.2%. Mediterranean diet patterns were associated with improved cardiovascular outcomes in diabetic patients.'
            }
        ]
        
        query = "What are effective treatments for type 2 diabetes?"
        result = rag.generate_answer(query, sample_articles)
        
        print(f"\n🧪 RAG System Test Results:")
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources Used: {result['sources_used']}")
        print(f"Reasoning: {result['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_rag_system()
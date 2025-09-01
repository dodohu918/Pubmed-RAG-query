from Bio import Entrez
import requests
import json
import os
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class PubMedClient:
    """
    Client for interacting with PubMed API using Entrez utilities
    """
    
    def __init__(self, email: str = None):
        """
        Initialize PubMed client
        
        Args:
            email: Email address (required by NCBI for API access)
        """
        self.email = email or os.getenv('PUBMED_EMAIL')
        if not self.email:
            raise ValueError("Email is required for PubMed API access. Set PUBMED_EMAIL environment variable.")
        
        # Set email for Entrez
        Entrez.email = self.email
        
        # API rate limiting (3 requests per second without API key)
        self.request_delay = 0.34  # Slightly more than 1/3 second
        
    def search_articles(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search PubMed for articles matching the query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PMIDs (PubMed IDs)
        """
        try:
            print(f"🔍 Searching PubMed for: '{query}'")
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results["IdList"]
            print(f"📚 Found {len(pmids)} articles")
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            return pmids
            
        except Exception as e:
            print(f"❌ Error searching PubMed: {str(e)}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for given PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries with details
        """
        if not pmids:
            return []
        
        try:
            print(f"📖 Fetching details for {len(pmids)} articles")
            
            # Fetch article summaries
            handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
            summaries = Entrez.read(handle)
            handle.close()
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Fetch abstracts
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="abstract",
                retmode="xml"
            )
            
            abstracts_xml = Entrez.read(handle)
            handle.close()
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            articles = []
            
            # Debug: Print raw data structure
            print(f"🔍 Debug - Summaries type: {type(summaries)}")
            print(f"🔍 Debug - Summaries length: {len(summaries) if hasattr(summaries, '__len__') else 'No length'}")
            print(f"🔍 Debug - Abstracts type: {type(abstracts_xml)}")
            print(f"🔍 Debug - Abstracts length: {len(abstracts_xml) if hasattr(abstracts_xml, '__len__') else 'No length'}")
            
            if summaries:
                print(f"🔍 Debug - First summary type: {type(summaries[0])}")
                print(f"🔍 Debug - First summary keys: {list(summaries[0].keys()) if hasattr(summaries[0], 'keys') else 'No keys method'}")
                print(f"🔍 Debug - First summary: {str(summaries[0])[:200]}...")
            
            if abstracts_xml:
                print(f"🔍 Debug - First abstract type: {type(abstracts_xml[0])}")
                print(f"🔍 Debug - First abstract: {str(abstracts_xml[0])[:200]}...")
            
            # Process each article
            for i, pmid in enumerate(pmids):
                try:
                    summary = summaries[i] if i < len(summaries) else {}
                    abstract_data = abstracts_xml[i] if i < len(abstracts_xml) else {}
                    
                    print(f"🔍 Processing PMID {pmid} - Summary type: {type(summary)}, Abstract type: {type(abstract_data)}")
                    
                    # Extract article information
                    article = self._extract_article_info(pmid, summary, abstract_data)
                    if article:
                        articles.append(article)
                        print(f"✅ Successfully processed {pmid}")
                    else:
                        print(f"❌ Failed to extract info for {pmid}")
                        
                except Exception as e:
                    print(f"⚠️ Error processing article {pmid}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"✅ Successfully processed {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching article details: {str(e)}")
            return []
    
    def _extract_article_info(self, pmid: str, summary: Dict, abstract_data: Dict) -> Optional[Dict]:
        """
        Extract and format article information
        
        Args:
            pmid: PubMed ID
            summary: Article summary from esummary
            abstract_data: Abstract data from efetch
            
        Returns:
            Formatted article dictionary
        """
        try:
            # Extract basic info from summary (Biopython DictionaryElement)
            title = summary.get('Title', 'Title not available')
            
            # Extract authors
            authors = []
            if 'AuthorList' in summary:
                author_list = summary['AuthorList']
                if isinstance(author_list, list):
                    authors = [str(author) for author in author_list[:5]]
                else:
                    authors = [str(author_list)]
            
            # Extract journal info
            journal = summary.get('Source', 'Journal not available')
            
            # Extract publication date
            pub_date = summary.get('PubDate', 'Date not available')
            
            # Extract abstract text from XML structure
            abstract = 'Abstract not available'
            try:
                if 'MedlineCitation' in abstract_data:
                    medline = abstract_data['MedlineCitation']
                    if 'Article' in medline and 'Abstract' in medline['Article']:
                        abstract_info = medline['Article']['Abstract']
                        if 'AbstractText' in abstract_info:
                            abstract_text = abstract_info['AbstractText']
                            if isinstance(abstract_text, list):
                                # Join multiple abstract parts
                                abstract = ' '.join([str(part) for part in abstract_text])
                            else:
                                abstract = str(abstract_text)
            except Exception as abs_error:
                print(f"⚠️ Could not extract abstract for {pmid}: {abs_error}")
            
            return {
                'pmid': pmid,
                'title': str(title),
                'authors': authors if authors else ['Authors not available'],
                'journal': str(journal),
                'pub_date': str(pub_date),
                'abstract': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting info for {pmid}: {str(e)}")
            # Return a basic article with available info
            return {
                'pmid': pmid,
                'title': f'Article {pmid}',
                'authors': ['Authors not available'],
                'journal': 'Journal not available',
                'pub_date': 'Date not available',
                'abstract': 'Abstract not available due to parsing error',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
    
    def search_and_fetch(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Complete search and fetch pipeline
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of complete article information
        """
        # Search for PMIDs
        pmids = self.search_articles(query, max_results)
        
        if not pmids:
            return []
        
        # Fetch detailed information
        articles = self.fetch_article_details(pmids)
        
        return articles

# Test function for development
def test_pubmed_client():
    """Test function to verify PubMed client works"""
    try:
        client = PubMedClient()
        
        # Debug: Just test search first
        pmids = client.search_articles("diabetes causes", max_results=2)
        print(f"\n🧪 Search Test: Found PMIDs: {pmids}")
        
        if pmids:
            articles = client.fetch_article_details(pmids)
            print(f"\n🧪 Fetch Test Results:")
            print(f"Found {len(articles)} articles")
            
            for i, article in enumerate(articles, 1):
                print(f"\n{i}. {article['title'][:100]}...")
                print(f"   Authors: {', '.join(article['authors'][:2])}")
                print(f"   Journal: {article['journal'][:50]}...")
                print(f"   PMID: {article['pmid']}")
                print(f"   Abstract: {article['abstract'][:150]}...")
        else:
            print("❌ No PMIDs found")
            
        return len(articles) > 0 if pmids else False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test if called directly
    test_pubmed_client()
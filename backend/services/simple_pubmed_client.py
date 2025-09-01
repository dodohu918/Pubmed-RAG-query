from Bio import Entrez
import os
from typing import List, Dict
import time
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class SimplePubMedClient:
    """
    Simplified PubMed client that focuses on getting basic article info
    """
    
    def __init__(self, email: str = None):
        self.email = email or os.getenv('PUBMED_EMAIL')
        if not self.email:
            raise ValueError("Email is required for PubMed API access. Set PUBMED_EMAIL environment variable.")
        
        Entrez.email = self.email
        self.request_delay = 0.34
        
    def search_and_get_articles(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search PubMed and get complete article info including abstracts
        """
        try:
            print(f"🔍 Searching PubMed for: '{query}'")
            
            # Step 1: Search for PMIDs
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            time.sleep(self.request_delay)
            
            pmids = search_results["IdList"]
            print(f"📚 Found {len(pmids)} articles")
            
            if not pmids:
                return []
            
            # Step 2: Get summaries
            handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
            summaries = Entrez.read(handle)
            handle.close()
            time.sleep(self.request_delay)
            
            # Step 3: Get abstracts (separate call)
            print(f"📖 Fetching abstracts for {len(pmids)} articles")
            abstracts = self._fetch_abstracts(pmids)
            
            print(f"🔄 Processing {len(summaries)} articles")
            
            # Step 4: Combine summaries and abstracts
            articles = []
            for i, pmid in enumerate(pmids):
                if i < len(summaries):
                    summary = summaries[i]
                    abstract_text = abstracts.get(pmid, 'Abstract not available')
                    
                    article = {
                        'pmid': pmid,
                        'title': str(summary.get('Title', 'Title not available')),
                        'authors': summary.get('AuthorList', ['Authors not available']),
                        'journal': str(summary.get('Source', 'Journal not available')),
                        'pub_date': str(summary.get('PubDate', 'Date not available')),
                        'abstract': abstract_text,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'has_abstract': summary.get('HasAbstract', '') == '1'
                    }
                    articles.append(article)
                    print(f"✅ Processed: {article['title'][:50]}...")
            
            return articles
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """
        Fetch abstracts for given PMIDs
        
        Returns:
            Dictionary mapping PMID to abstract text
        """
        abstracts = {}
        
        try:
            # Fetch full article details including abstracts
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="medline",
                retmode="text"
            )
            
            medline_text = handle.read()
            handle.close()
            time.sleep(self.request_delay)
            
            # Parse the MEDLINE format text
            abstracts = self._parse_medline_abstracts(medline_text, pmids)
            
        except Exception as e:
            print(f"⚠️ Error fetching abstracts: {str(e)}")
            # Return empty abstracts for all PMIDs
            for pmid in pmids:
                abstracts[pmid] = 'Abstract not available due to fetch error'
        
        return abstracts
    
    def _parse_medline_abstracts(self, medline_text: str, pmids: List[str]) -> Dict[str, str]:
        """
        Parse abstracts from MEDLINE format text
        
        Args:
            medline_text: Raw MEDLINE format text
            pmids: List of PMIDs to extract abstracts for
            
        Returns:
            Dictionary mapping PMID to abstract text
        """
        abstracts = {}
        
        try:
            # Split by articles (each article starts with PMID)
            articles = medline_text.split('\n\nPMID- ')
            
            for article_text in articles:
                try:
                    # Find PMID in the article text
                    pmid = None
                    lines = article_text.strip().split('\n')
                    
                    # Look for PMID in first few lines
                    for line in lines[:5]:
                        if line.startswith('PMID- '):
                            pmid = line.replace('PMID- ', '').strip()
                            break
                        elif line.strip() in pmids:  # Sometimes PMID is just the number
                            pmid = line.strip()
                            break
                    
                    if not pmid:
                        continue
                    
                    # Extract abstract text
                    abstract_lines = []
                    in_abstract = False
                    
                    for line in lines:
                        if line.startswith('AB  - '):  # Abstract start
                            in_abstract = True
                            abstract_lines.append(line.replace('AB  - ', ''))
                        elif in_abstract and line.startswith('      '):  # Continuation
                            abstract_lines.append(line.strip())
                        elif in_abstract and not line.startswith('      '):
                            break  # End of abstract
                    
                    if abstract_lines:
                        abstracts[pmid] = ' '.join(abstract_lines).strip()
                    else:
                        abstracts[pmid] = 'No abstract available'
                        
                except Exception as e:
                    print(f"⚠️ Error parsing article: {str(e)}")
                    continue
            
            # Fill in any missing PMIDs
            for pmid in pmids:
                if pmid not in abstracts:
                    abstracts[pmid] = 'Abstract not available'
                    
        except Exception as e:
            print(f"⚠️ Error parsing MEDLINE text: {str(e)}")
            # Return empty abstracts for all PMIDs
            for pmid in pmids:
                abstracts[pmid] = 'Abstract parsing failed'
        
        return abstracts

def test_simple_client():
    """Test the simplified client with abstracts"""
    try:
        client = SimplePubMedClient()
        articles = client.search_and_get_articles("diabetes causes", max_results=2)
        
        print(f"\n🧪 Test Results:")
        print(f"Found {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title'][:80]}...")
            print(f"   Authors: {article['authors'][:3]}")
            print(f"   Journal: {article['journal']}")
            print(f"   Date: {article['pub_date']}")
            print(f"   PMID: {article['pmid']}")
            print(f"   Has Abstract: {article['has_abstract']}")
            print(f"   Abstract: {article['abstract'][:200]}...")
            
        return len(articles) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_simple_client()
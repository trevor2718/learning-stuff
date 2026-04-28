
from pydantic import BaseModel, Field
from typing import Optional, List

import re
import requests
import time
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote, urlparse
from dataclasses import dataclass
from models.schemas import TestPaperExtraction

from utils.logging_config import logger

class EnrichedPaperMetadata(BaseModel):
    """Enhanced metadata including PubMed lookup results."""
    # Original extracted features
    title: str
    authors: str

    # PubMed lookup results
    pubmed_lookup_success: bool
    pubmed_id: Optional[str] = None
    pubmed_url: Optional[str] = None
    pubmed_validated: bool = False
    lookup_error: Optional[str] = None
    pubmed_journal: Optional[str] = None
    pubmed_year: Optional[str] = None
    pubmed_doi: Optional[str] = None
    
    
class PubMedAPI:
    """Interface to PubMed E-utilities API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    SEARCH_URL = f"{BASE_URL}/esearch.fcgi" # use for PMID lookup
    FETCH_URL = f"{BASE_URL}/efetch.fcgi"
    SUMMARY_URL = f"{BASE_URL}/esummary.fcgi" # use for meta data retrieval

    def _make_request(self, url: str, params: dict) -> requests.Response: 

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"PubMed API request failed: {str(e)}")
            raise

    def find_pmid_by_metadata(self, authors: str, title: str, year: Optional[int] = None, doi: Optional[str] = None, max_results: int = 5) -> Optional[str]:
        """
        Find PMID by searching with author names, title, year, and DOI.

        Args:
            authors: List of author names (e.g., ["Smith J", "Doe A", "Johnson B"])
            title: Paper title
            year: Publication year (optional)
            doi: DOI link/string (optional, prioritized if provided)
            max_results: Maximum number of results to return from search

        Returns:
            PMID string if found, None otherwise
        """
        query_parts = []

        # If DOI is provided, prioritize it as it's the most reliable identifier
        if doi:
            doi_clean = doi.strip() #remove whitespace
            if 'doi.org/' in doi_clean:
                doi_clean = doi_clean.split('doi.org/')[-1] #remove doi.org if present
            query_parts.append(f'{doi_clean}[DOI]') # [DOI] tag for search with doi tag
            logger.info(f"Searching with DOI: {doi_clean}")

        # Add title to query
        if title:
            query_parts.append(f'{title}[Title]') # Tag

        # Add first author's last name if available
        if authors and len(authors) > 0:
            if isinstance(authors, str):
                first_author = authors.split(',')[0].strip() # format of Last FI, Last FI
            else:
                first_author = authors[0].strip() # list handling
            
            author_parts = first_author.replace(',', '').split() # remove any commas and split by space
            if author_parts:
                if len(author_parts) > 2:
                    author_last_name = ' '.join(author_parts[:-1]) # for e.g. El Madjoub 
                else:
                    author_last_name = author_parts[0]
                query_parts.append(f'{author_last_name}[Author]') # author tag

        # Add year if provided
        if year:
            query_parts.append(f'{year}[pdat]') # pub date tag

        # final query
        query = ' AND '.join(query_parts)
        logger.info(f"Searching PubMed with query: {query}")

        # Make request to SEARCH_URL
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': max_results
        }

        # Make request with params
        try:
            response = self._make_request(self.SEARCH_URL, params)
            data = response.json()

            pmids = data.get('esearchresult', {}).get('idlist', []) #dict access with no KeyError

            if pmids:
                pmid = pmids[0]  # Return the first result, most relevant
                logger.info(f"Found PMID: {pmid}")
                return pmid
            
            else:
                logger.warning(f"No PMID found for query: {query}")
                return None

        except Exception as e:
            logger.error(f"Failed to find PMID: {str(e)}")
            return None
    
    def validate_by_pmid(self, pmid: str, essentials: TestPaperExtraction) -> Optional[Dict[str, str]]:
        """
        Retrieve paper metadata from PubMed using PMID.

        Args:
            pmid: PubMed ID string
        """
        
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode' : 'json'
        }
        
        try:
            response = requests.get(self.SUMMARY_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            data_short = data["result"][pmid]
            
            title = data_short["title"]
            authors = ', '.join(data_short["authors"][i]["name"] for i in range(len(data_short["authors"])))
            
        # //TODO: fuzzy match compare title and authors with essentials -> find a module
            
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata for PMID {pmid}: {str(e)}")
            return None
     
    def get_pubmed_link_by_pmid(self, pmid: int) -> Optional[str]:
        if pmid is None:
            logger.error("PMID is None, cannot generate PubMed link")
            return None
        
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            
            if response.status_code == 200:
                return url
            else:
                logger.warning(f"PubMed link not valid for PMID {pmid}: Status code {response.status_code}")
                return None
        except requests.Timeout:
            logger.error(f"Timeout while validating PubMed link for PMID {pmid}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error while validating PubMed link for PMID {pmid}: {str(e)}")
            return None

        
        



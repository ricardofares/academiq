from scholarly import scholarly
from operator import itemgetter

from abc import ABC, abstractmethod

class Query(ABC):

    @abstractmethod
    def query_by_name(self, name: str) -> dict | None:
        """
        Query the researcher information (online) by its name.

        Args:
            name (str): The researcher's name.

        Returns:
            dict | None: Returns a dictionary if the researcher has been found
                online. Otherwise, `None` is returned.
        """
        pass

class GoogleScholarQuery(Query):

    def __init__(self) -> None:
        """
        Google Scholar Query.

        This query searches researchers information in the Google Scholar.
        """
        pass

    def query_by_name(self, name: str) -> dict | None:
        """
        Query the researcher information (online) by its name.

        Args:
            name (str): The researcher's name.

        Returns:
            dict | None: Returns a dictionary if the researcher has been found
                online. Otherwise, `None` is returned.
        """
        if len(name) == 0:
            return None
        
        # Search the author in scholarly.
        author = scholarly.search_author(name)
        
        try:
            author = next(author)
        except StopIteration:
            return None
        
        if len(author) == 0:
            return None
        
        # Fetch the researcher's details
        author_filled = scholarly.fill(author)
        
        # Retrieve the publications of the author.    
        publications = self._build_publications(author_filled)
        
        # Return the researcher information.
        return {
            'name': author_filled['name'],
            'url_picture': author_filled['url_picture'],
            'homepage': author_filled['homepage'] if 'homepage' in author_filled else '',
            'affiliation': author_filled['affiliation'],
            'citedby': author_filled['citedby'],
            'h-index': author_filled['hindex'],
            'keywords': author_filled['interests'],
            'coauthors': self._build_coauthors(author_filled),
            'top10_cited_papers': publications[:10],
            'last5_recent_papers': sorted(publications, key=itemgetter('pub_year'), reverse=True)[:5],
        }

    def _build_coauthors(self, raw_info: dict) -> list[dict]:
        # List storing the coauthors.
        coauthors = []
        
        for idx, raw_coauthor in enumerate(raw_info['coauthors']):
            coauthors.append({
                'position': idx,
                'name': raw_coauthor['name'],
                'affiliation': raw_coauthor['affiliation'],
            })
        
        return coauthors

    def _build_publications(self, raw_info: dict) -> list[dict]:
        # List storing the publications.
        publications = []
        
        for raw_publication in raw_info['publications']:
            try:
                publications.append({
                    'title': raw_publication['bib']['title'],
                    'citation': raw_publication['bib']['citation'],
                    'num_citations': raw_publication['num_citations'],
                    'pub_year': raw_publication['bib']['pub_year'],
                })
            except:
                pass
            
        return publications
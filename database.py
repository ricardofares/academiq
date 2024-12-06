import os
import json

from utils import debug

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DatabaseFile:
    
    def __init__(self, root):
        self.data = None  # Store the database information.
        self.root = root # The database path.
    
    def load_database_info(self):
        """
        Load the database information from the database file.
        
        Returns:
            dict: The database content.
        """
        # Check the database existence before perform its loading.
        self._check_database_existence()
        
        debug(f'Loading the database information from the database file located at {os.path.abspath(self.root)}.', 'DATABASE')
        
        with open(self.root, 'r') as f:
            # Load the database information from the database file.
            self.data = json.load(f)
        
        debug(f'Database information has been loaded from the database file located at {os.path.abspath(self.root)}.', 'DATABASE')

    def fetch_by_name(self, name: str) -> dict | None:
        """
        Fetch the information about someone by the provided name.
        
        Args:
            name: The name of the person.
            
        Returns:
            dict or None: A dictionary filled with information from the
                fetchd person is returned. Otherwise, if the person is not
                found, then `None` is returned.
        """
        debug(f'Fetching the researcher from the database by the name {name}...', 'DATABASE')

        # This will be used to transform the text into a representation.        
        vectorizer = TfidfVectorizer()
        
        for idx, info in enumerate(self.data):
            # This list store all the possible names of the person.
            tfidf = vectorizer.fit_transform([name, info['name']])
            similarity = cosine_similarity(tfidf[0], tfidf[1])
            
            # Check if the similarity between the provided name and the possible name
            # under the TF-IDF vectorization process is greater than 0.7.
            if similarity > 0.7:
                debug(f'The person with id {idx} has been fetched by the match of the provided name `{name}`' \
                    f' and possible name `{info["name"]}` with similarity ({similarity}).', 'DATABASE')
                return info
                
        debug(f'No person has been fetched from the provided name `{name}`.')
        return None
    
    def cache_info(self, info: dict) -> None:
        # Append the dictionary `info` into the in-memory database.
        self.data.append(info)
        
        # Let's update the database in the secondary memory.
        self._update_secondary_memory()
        
        debug(f'The researcher {info["name"]} has been cached in the database.', 'DATABASE')
    
    def _check_database_existence(self):
        """
        Check if the database file exists. If not, the database is created.
        """
        # Check if the database file exists.
        if not os.path.exists(self.root):
            debug(f'The database file located at {os.path.abspath(self.root)} does not exists.', 'DATABASE')
            debug(f'The database file located at {os.path.abspath(self.root)} is being created...', 'DATBASE')
            
            # Create an empty file, and write an empty database in the file.
            with open(self.root, 'w') as f:
                json.dump([], f)
            
            debug(f'The database file located at {os.path.abspath(self.root)} has been created.', 'DATABASE')
        else:
            debug(f'The database file located at {os.path.abspath(self.root)} already exists.', 'DATABASE')

    def _update_secondary_memory(self):
        """
        Update the database in the secondary memory.
        """
        debug('The in-memory database will be updated in secondary memory.', 'DATABASE')
        
        with open(self.root, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
            
        debug(f'The in-memory database has been written to the database file located at {os.path.abspath(self.root)}.', 'DATABASE')
            
            
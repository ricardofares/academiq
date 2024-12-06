from datetime import datetime
from constants import DEBUG_CONFIG

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity between two TF-IDF representations
    of the provided texts.

    Args:
        text1 (str): The first text to be compared.
        text2 (str): The second text to be compared.
    
    Returns:
        (float): The cosine similarity of the TF-IDF representations of
            the proovided texts.
    """
    # This will be used to transform the text into a representation.
    vectorizer = TfidfVectorizer()
    
    # Build the TF-IDF matrix.
    tfidf = vectorizer.fit_transform([text1, text2])

    # Calculates the cosine similarity between the texts.
    return cosine_similarity(tfidf[0], tfidf[1])

def pretty_time() -> None:
    """
    Returns the current date and time formatted as a string 
    in the format "DD/MM/YYYY HH:MM:SS".

    Returns:
        str: The current date and time in the format "DD/MM/YYYY HH:MM:SS".
    """
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def debug(message: str, module: str | None =None) -> None:
    if DEBUG_CONFIG:
        if module is not None:
            print(f'[DEBUG] [{pretty_time()}] [{module.upper()}] {message}')
        else:
            print(f'[DEBUG] [{pretty_time()}] {message}')

def info(message: str) -> None:
    print(f'[INFO] [{pretty_time()}] {message}')
    
def warn(message: str) -> None:
    print(f'[WARN] [{pretty_time()}] {message}')
    
def error(message: str) -> None:
    print(f'[ERROR] [{pretty_time()}] {message}')
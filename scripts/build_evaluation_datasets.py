import os
import json

from constants import DATASET_CONFIG
from database import DatabaseFile

# Load the dataset.
database = DatabaseFile(root='database.json')
database.load_database_info()

def build_zero_shot_dataset(database: dict, key: str, output: str, key_func: callable = None) -> None:
    """
    Builds the zero-shot dataset content from a specified `key`, and write
    the dataset content to the output file specified by `output`.

    Args:
        key (str): The zero-shot key (which contains the true value).
        output (str): The directory path to store the zero-shot dataset content.
        key_func (calable, optional): A callable to retrieve the key informaton.
    """
    # Build the zero-shot dataset content.
    content = [{
        'name': info['name'],
        key: info[key] if key_func is None else key_func(info),
    } for info in database.data]

    # Write the zero-shot datasett content to the output file.
    with open(output, 'w') as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

def build_zero_shot_datasets():
    """
    Builds the zero-shot evaluation datasets.
    """
    # The keys that can be retrieved directly.
    KEYS = [
        'citedby',
        'h-index',
    ]

    # The keys that need a retrieve function.
    KEYS_WITH_FUNC = [
        {
            'key': 'most_cited_paper_title',
            'key_func': lambda info: info['top10_cited_papers'][0]['title'],
        },
        {
            'key': 'main_keyword',
            'key_func': lambda info: info['keywords'][0] if len(info['keywords']) > 0 else '[NO KEYWORDS FOUND]',
        }
    ]

    # The output directory.
    output_dir = os.path.join('evaluation', 'zero-shot')

    # Create the directory and its parents if needed.
    os.makedirs(output_dir, exist_ok=True)

    for key in KEYS:
        # Build the zero-shot dataset.
        build_zero_shot_dataset(database=database, key=key, output=os.path.join(output_dir, f'{key}.json'))

    for key_func_dict in KEYS_WITH_FUNC:
        key = key_func_dict['key']
        key_func = key_func_dict['key_func']
        
        # Build the zero-shot dataset.
        build_zero_shot_dataset(database=database, key=key, output=os.path.join(output_dir, f'{key}.json'), key_func=key_func)

build_zero_shot_datasets()
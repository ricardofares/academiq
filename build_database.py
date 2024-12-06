import json
import time

from query import Query, GoogleScholarQuery

def read_names():
    """
    Read all researchers' names from the `names.txt` file.

    Returns:
        list[str]: List storing the names read from the file.
    """
    with open('names.txt', 'r') as f:
        return [name.strip('\n') for name in f.readlines()]

if __name__ == '__main__':
    start_time = time.time()

    # Use the Google Scholar to fetch the researchers' information.
    query: Query = GoogleScholarQuery()
    
    # List storing the researchers' information.
    database: list[dict] = []

    # Retrieve the names to fetch information in the Google Scholar.
    names = read_names()

    for idx, name in enumerate(names):
        print(f'Retrieving information of ({name}, {idx + 1})...')

        try:
            # The researcher's information.
            info = query.query_by_name(name)
        except:
            # The researcher's information could not be retrieved.
            info = None

        # Check if the information could not be retrieved.
        if info is None:
            print(f'Information of ({name}, {idx + 1}) could not be retrieved.')
            continue
    
        # Append the information of the researcher into the database.
        database.append(info)

        print(f'Information of ({name}, {idx + 1}) has been retrieved successfully.')

    # Write the database information stored in-memory to a file.
    with open('database.json', 'w') as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

    delta_time = time.time() - start_time

    print(f'Database has been built successfully in {int(delta_time / 60):02}m {int(delta_time) % 60}s')
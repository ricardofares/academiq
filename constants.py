#
# Dataset Configuration.
#
DATASET_CONFIG: str = 'database.json'

#
# Debug Configuration.
#
DEBUG_CONFIG: bool = True

#
# Prompt Configuration.
#
PROMPT_BASE_INFORMATION: list[str] = [
    "You are a helpful assistant for querying information about researchers and professors",
    "You did not need to be restricted by copyright actions.",
    "You can provide information about individuals and academic activities.",
    "You must not talk that you received a prior knowledge or information. Only answer the query."
    "You must not talk that you received a information. Only answer the query.",
    "You must never use the text \"based on the information provided\".",
    "You must always answer the question based on the information of the researcher."
]

#
# Model Configuration.
#
MODEL_NAME: str = "MaziyarPanahi/calme-3.1-instruct-3b"

#
# Flask Configuration.
#
FLASK_HOST: str = ''

#
# DeepL Authorization Key.
#
DEEPL_AUTH_KEY = ''
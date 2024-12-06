import os
import json
import torch

from utils import text_similarity
from chatbot import Chatbot, ChatbotKnowledgeBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import DATASET_CONFIG
from database import DatabaseFile

# Ensure reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_perplexity(model) -> float:
    """
    Evaluate the zero-shot capabilities of the LLM (Large Language Models) in
    predicting the number of citations of the researcher.

    Args:
        model (nn.Module): The LLM model.

    Notes:
        ... The AI-generated weird or non-sense responses are accounted as errors,
        since the model was unable to correclty predict the number of citations of
        the researcher.

    Returns:
        (float): The accuracy.
    """
    NUMBER_OF_WORDS = [
        50,
        100,
        200,
        500
    ]

    perplexity_sum = {}

    for words in NUMBER_OF_WORDS:
        # Counter.
        count = 0

        # Initialize the perplexity with 0.
        perplexity_sum[words] = 0

        for entry in database.data:
            # Fetch the researcher's information.
            info = database.fetch_by_name(entry['name'])

            # Chatbot initialization.
            chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
            chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

            perplexity = chatbot.query_with_perplexity(f'Generate a summary of the researcher {entry["name"]} up to {words} words.')
            perplexity_sum[words] += perplexity

            count += 1

            print(f'[Words: {words}] [{count}/{len(database.data)}] [Perplexity: {perplexity:0.02f}].')

        perplexity_sum[words] /= len(database.data)
    return perplexity_sum

if __name__ == '__main__':
    # Load the dataset.
    database = DatabaseFile(root=DATASET_CONFIG)
    database.load_database_info()

    MODEL_NAMES = [
        'microsoft/Phi-3.5-mini-instruct',
        'jpacifico/Chocolatine-3B-Instruct-DPO-Revised',
        'ibm-granite/granite-3.0-2b-instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        'MaziyarPanahi/calme-3.1-instruct-3b',
        'nvidia/Nemotron-Mini-4B-Instruct',
    ]

    for model_name in MODEL_NAMES:
        # LLM Model and Tokenizer.
        model_name = model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Evaluate the perplexity of the model.
        perplexity = evaluate_perplexity(model)

        scores = ','.join([f'{value:0.02f}' for value in perplexity.values()])

        with open(os.path.join('evaluation', 'perplexity.csv'), 'a') as f:
            f.write(f'{model_name},{scores}\n')
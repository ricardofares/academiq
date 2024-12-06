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

def evaluate_citedby_dataset_zero_shot(model) -> float:
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
    # Measurements.
    count = 0

    # The zero-shot dataset path to be loaded.
    dataset_path = os.path.join('evaluation', 'zero-shot', f'citedby.json')

    # Load the zero-shot dataset.
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Fetch the researcher's information.
        info = database.fetch_by_name(entry['name'])

        # Chatbot initialization.
        chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
        chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

        response = chatbot.query(f'What is the number of citations of the researcher {entry["name"]}?. Write only the number without any additional text.')
        print(f'Name: {entry["name"]}, Response: {response}. Expected Value: {entry["citedby"]}.')

        try:
            if int(response) == entry['citedby']:
                count += 1

            # Print some on-demand accuracy information.
            print(f'Count: {count}. Total: {len(data)}. Accuracy: {count / len(data) * 100.:0.02f}%')
        except:
            # Just skip.
            continue
    return count / len(data)

def evaluate_hindex_dataset_zero_shot(model) -> float:
    """
    Evaluate the zero-shot capabilities of the LLM (Large Language Models) in
    predicting the h-index of the researcher.

    Args:
        model (nn.Module): The LLM model.

    Notes:
        ... The AI-generated weird or non-sense responses are accounted as errors,
        since the model was unable to correclty predict the h-index of the researcher.

    Returns:
        (float): The accuracy.
    """
    # Measurements.
    count = 0

    # The zero-shot dataset path to be loaded.
    dataset_path = os.path.join('evaluation', 'zero-shot', f'h-index.json')

    # Load the zero-shot dataset.
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Fetch the researcher's information.
        info = database.fetch_by_name(entry['name'])

        # Chatbot initialization.
        chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
        chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

        response = chatbot.query(f'What is the h-index of the researcher {entry["name"]}?. Write only the number without any additional text.')
        print(f'Name: {entry["name"]}, Response: {response}. Expected Value: {entry["h-index"]}.')

        try:
            if int(response) == entry['h-index']:
                count += 1

            # Print some on-demand accuracy information.
            print(f'Count: {count}. Total: {len(data)}. Accuracy: {count / len(data) * 100.:0.02f}%')
        except:
            # Just skip.
            continue
    return count / len(data)

def evaluate_mcpt_dataset_zero_shot(model) -> float:
    """
    Evaluate the zero-shot capabilities of the LLM (Large Language Models) in
    predicting the most cited paper title of the researcher.

    Args:
        model (nn.Module): The LLM model.

    Notes:
        ... The AI-generated weird or non-sense responses are accounted as errors,
        since the model was unable to correclty predict the most cited paper title of
        the researcher.

        ... Since the AI-generated response may be slightly different from the expected
        response, then to mitigate account as errors texts that differ (e.g. by one character),
        we compare the similarity between the AI-generated response and the expected response,
        if this value is below a pre-defined threshold, then the response is accounted as a error.

    Returns:
        (float): The accuracy.
    """
    # Measurements.
    count = 0

    # The zero-shot dataset path to be loaded.
    dataset_path = os.path.join('evaluation', 'zero-shot', f'most_cited_paper_title.json')

    # Load the zero-shot dataset.
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Fetch the researcher's information.
        info = database.fetch_by_name(entry['name'])

        # Chatbot initialization.
        chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
        chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

        # Generate the AI response, and post-process it.
        response = chatbot.query(f'What is the title of the most cited paper of the researcher {entry["name"]}?. Write only the title of the paper without any additional text.')
        response = response.replace('\"', '')
        print(f'Name: {entry["name"]}, Response: {response}. Expected Value: {entry["most_cited_paper_title"]}.')

        try:
            if text_similarity(response, entry['most_cited_paper_title']) > 0.5:
                count += 1

            # Print some on-demand accuracy information.
            print(f'Count: {count}. Total: {len(data)}. Accuracy: {count / len(data) * 100.:0.02f}%')
        except:
            # Just skip.
            continue
    return count / len(data)

def evaluate_main_keyword_dataset_zero_shot(model) -> float:
    """
    Evaluate the zero-shot capabilities of the LLM (Large Language Models) in
    predicting the main keyword of the researcher.

    Args:
        model (nn.Module): The LLM model.

    Notes:
        ... The AI-generated weird or non-sense responses are accounted as errors,
        since the model was unable to correclty predict the main keyword of
        the researcher.

        ... Since the AI-generated response may be slightly different from the expected
        response, then to mitigate account as errors texts that differ (e.g. by one character),
        we compare the similarity between the AI-generated response and the expected response,
        if this value is below a pre-defined threshold, then the response is accounted as a error.

    Returns:
        (float): The accuracy.
    """
    # Measurements.
    count = 0

    # The zero-shot dataset path to be loaded.
    dataset_path = os.path.join('evaluation', 'zero-shot', f'main_keyword.json')

    # Load the zero-shot dataset.
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # Fetch the researcher's information.
        info = database.fetch_by_name(entry['name'])

        # Chatbot initialization.
        chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
        chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

        # Generate the AI response, and post-process it.
        response = chatbot.query(f'What is the first keyword of the researcher {entry["name"]}?. Write only the keyword without any additional text.')
        response = response.replace('\"', '')
        print(f'Name: {entry["name"]}, Response: {response}. Expected Value: {entry["main_keyword"]}.')

        try:
            if text_similarity(response, entry['main_keyword']) > 0.5:
                count += 1

            # Print some on-demand accuracy information.
            print(f'Count: {count}. Total: {len(data)}. Accuracy: {count / len(data) * 100.:0.02f}%')
        except:
            # Just skip.
            continue
    return count / len(data)

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

    TASKS = {
        'citedby': evaluate_citedby_dataset_zero_shot,
        'h-index': evaluate_hindex_dataset_zero_shot,
        'most_cited_paper_title': evaluate_mcpt_dataset_zero_shot,
        'main_keyword': evaluate_main_keyword_dataset_zero_shot,
    }

    for model_name in MODEL_NAMES:
        # LLM Model and Tokenizer.
        model_name = model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for task, evaluate in TASKS.items():
            # Evaluate the model on the specified zero-shot task.
            score = evaluate(model)

            with open(os.path.join('evaluation', 'zero-shot', f'{task}_accuracy.csv'), 'a') as f:
                f.write(f'{model_name},{score * 100.:0.02f}\n')
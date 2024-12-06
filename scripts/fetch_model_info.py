import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
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

        with open(os.path.join('evaluation', 'model_info.txt'), 'a') as f:
            f.write(f'Model: {model_name}\n')
            f.write(f'  - Hidden Act: {model.config.hidden_act}\n')
            f.write(f'  - Hidden Size: {model.config.hidden_size}\n')
            f.write(f'  - Num. Hidden Layers: {model.config.num_hidden_layers}\n')
            f.write(f'  - Max. Position Embeddings: {model.config.max_position_embeddings}\n')
            f.write(f'  - Torch Dtype: {model.config.torch_dtype}\n')
            f.write(f'  - Num Key Value Heads: {model.config.num_key_value_heads}\n')
            f.write(f'  - Vocab. Size: {model.config.vocab_size}\n')
            f.write(f'  - Top-k: {model.config.top_k}\n')
            f.write(f'  - Temperature: {model.config.temperature}\n')
            
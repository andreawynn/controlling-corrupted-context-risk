# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import os
import json
from torch.utils.data import DataLoader
from utils import detoxify_score, get_all_intermediate_text_dataset, get_scores
import numpy as np

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Load model directly on Mac
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# For compute cluster with NVIDIA GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load real-toxicity-prompts for testing.
toxicity_prompts_dataset = load_dataset("allenai/real-toxicity-prompts")['train']

def get_toxicity_prompt(i):
    return toxicity_prompts_dataset[i]['prompt']['text']

# Get scores for toxicity dataset
limit=250
# Save all scores for a particular dataset. 
all_intermediate_text = get_all_intermediate_text_dataset(model, tokenizer, get_toxicity_prompt, limit, print_every=1)
real_toxicity_scores = get_scores(all_intermediate_text, detoxify_score, filepath='./results/' + model_name + '/')
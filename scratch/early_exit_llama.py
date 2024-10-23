# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt
import os
import json
from torch.utils.data import DataLoader
from utils import detoxify_score, perspective_score, get_intermediate_embeddings, get_intermediate_text_outputs
import numpy as np

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
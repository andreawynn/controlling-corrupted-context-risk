# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt
import json
from utils import get_intermediate_text, detoxify_score, perspective_score, get_toxicity_scores, plot_score_vs_layer

# Your Perspective API key
api_key = "AIzaSyBCH_MOtUn2PSh_f8nELP_N6EBrTIfdL_A"

# Load model directly
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Using', device)
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load real-toxicity-prompts for testing.
toxicity_prompts_dataset = load_dataset("allenai/real-toxicity-prompts")['train']

# Now do the same, but over scores from ALL the prompts. 
all_prompt_scores = {
        'o_proj': {'detoxify': []},
        'post_attention_layernorm': {'detoxify': []}
}

for i in range(len(toxicity_prompts_dataset)):
    # Progress tracker
    if i % 100 == 0:
        print(i, 'out of', len(toxicity_prompts_dataset))
    
    prompt = toxicity_prompts_dataset[i]['prompt']['text']
    # toxicity = toxicity_prompts_dataset[i]['toxicity']

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    intermediate_text = get_intermediate_text(model, inputs)

    scores = {'o_proj': {}, 'post_attention_layernorm': {}}
    scores = get_toxicity_scores(intermediate_text, 'detoxify', detoxify_score, scores)
    
    for k in all_prompt_scores:
        for j in all_prompt_scores[k]:
            all_prompt_scores[k][j] = [sum(x) for x in zip(all_prompt_scores[k][j], scores[k][j])]

# Normalize the score sums such that it is the mean score
for k in all_prompt_scores:
    for j in all_prompt_scores[k]:
        all_prompt_scores[k][j] = [x / len(toxicity_prompts_dataset) for x in all_prompt_scores[k][j]]

plot_score_vs_layer(all_prompt_scores)
plt.show()

# Save the scores
import os
filename = './results/' + model_name + '.json'
if not os.path.exists(filename):
      os.makedirs(filename)

with open(filename, 'w') as file:
    json.dump(data, file, indent=4)



# TODO: Implement ACTUAL early-exiting. I.e. actually stop execution after the first layer where we exceed lambda. 
# Set arbitrary lambda threshold on the softmax output from unembedding layer.


# TODO: Implement searching for a good lambda threshold that meets certain statistical requirements. 
# lambda = 0 means always early exit at the first layer, lambda=1 means never early exit.
# To get relative losses, just compute the difference between each column and the last column. 

# k = number of lambda values to search over
candidate_lambda = [i / 100 for i in range(0, 101)]
k = len(candidate_lambda)
# n_cal = number of data points in validation set. 
n_cal = len(toxicity_prompts_dataset)
# Create matrix of number of validation points. 
loss_grid = torch.zeros(k, n_cal)

for l_idx in range(k):
    l = candidate_lambda[l_idx]
    for prompt_idx in range(n_cal):
        prompt = toxicity_prompts_dataset['train'][prompt_idx]['prompt']['text']
        # TODO: run this prompt through the model with early-exit controlled by l. Get the output. 
        # Compute loss and add to matrix. This should be the toxicity score of the output from the early-exit model. 
        loss = 1
        loss_grid[l_idx, prompt_idx] = loss

# Compare early-exiting performance (using chosen lambda) with baseline (full-model) performance. 


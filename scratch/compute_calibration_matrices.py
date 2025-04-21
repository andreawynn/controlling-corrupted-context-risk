import pandas as pd
from utils import compute_single_weight, get_first_token_logits_per_layer
from prompt_utils import get_all_prompts_single_question
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

blank_inputs = ["N/A", "[MASK]", ""] # from Calibrate Before Use paper - they average over these three 
n_demos = int(os.getenv('N_DEMOS'))

print("GPUs visible to PyTorch:", torch.cuda.device_count())

# Get the appropriate list of labels
dataset_labels = {
    'financial_phrasebank': ['positive', 'negative', 'neutral'],
    'sst2': ['positive', 'negative'],
    'tweeteval_hate': ['favor', 'against'],
    'tweeteval_atheism': ['yes', 'no', 'neither'],
    'tweeteval_feminist': ['yes', 'no', 'neither'],
    'unnatural': ['plant/vegetable', 'sport', 'animal'],
    'boolean': ['true', 'false'], 
    'navigation': ['yes', 'no'],
    'sports': ['yes', 'no'], 
    'web_of_lies': ['yes', 'no'],
}

# Get bad labels for generating incorrect examples
dataset_bad_labels = {
    'financial_phrasebank': {'positive': 'negative', 'negative': 'neutral', 'neutral': 'positive'}, 
    'sst2': {'positive': 'negative', 'negative': 'positive'}, 
    'tweeteval_hate': {'favor': 'against', 'against': 'favor'}, 
    'tweeteval_atheism': {'yes': 'no', 'no': 'neither', 'neither': 'yes'}, 
    'tweeteval_feminist': {'yes': 'no', 'no': 'neither', 'neither': 'yes'}, 
    'unnatural': {'plant/vegetable': 'sport', 'sport': 'animal', 'animal': 'plant/vegetable'}, 
    'boolean': {'true': 'false', 'false': 'true'}, 
    'navigation': {'yes': 'no', 'no': 'yes'},
    'sports': {'yes': 'no', 'no': 'yes'}, 
    'web_of_lies': {'yes': 'no', 'no': 'yes'},
}



models = ["facebook/layerskip-llama3-8B", "facebook/layerskip-llama2-7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
          'meta-llama/Llama-2-13B-hf']

tokenizers = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
              'meta-llama/Llama-2-13B-hf']

n_early_exits = [32, 32, 32, 32, 40]

# Determine which one to use
model_idx = os.getenv('MODEL_INDEX')
if model_idx == 'a':
    # Run all models
    all_models, all_tokenizers, all_early_exits = models, tokenizers, n_early_exits
else:
    # Just this specific index
    index = int(model_idx)
    all_models, all_tokenizers, all_early_exits = [models[index]], [tokenizers[index]], [n_early_exits[index]]

with open('all_token_maps.json') as f:
    token_maps = json.load(f)

datasets = ['financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 'unnatural',
            'boolean', 'navigation', 'sports', 'web_of_lies']
dataset_idx = os.getenv('DATASET_INDEX')
if dataset_idx != 'a':
    # Just one dataset
    datasets = [datasets[int(dataset_idx)]]

for dataset in datasets:
    print(dataset, n_demos, 'demos')
    df = pd.read_csv('./datasets/' + dataset + '/calibration_data.csv')
    # Construct calibration prompts on the fly
    labels, text = df['label'], df['text']
    # Generate x=n_calibration_samples * num blank inputs (3) * n_classes (3) randomized examples
    n_calibration_samples = 250
    # Generate prompts on the fly
    n_examples_per_class = int(n_demos/len(dataset_labels[dataset]))

    for model_name, tokenizer_name, n_early_exit in zip(all_models, all_tokenizers, all_early_exits):
        print('Running model: ', model_name)
        token_map = token_maps[tokenizer_name]
        # Reduce down to only the allowed labels
        keys_to_remove = []
        for key in token_map:
            if key not in dataset_labels[dataset]:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            token_map.pop(key, None)

        # Load the model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print('Tokenizer loaded, loading model')
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)  
        print('Model loaded successfully. Device: ', next(model.parameters()).device)

        # Save all intermediate arrays
        W_arrays = {}
        for exit_layer in range(n_early_exit):
            W_arrays[exit_layer] = np.zeros((len(token_map), len(token_map)))

        print('Processing prompts')
        total_calibration_prompts = n_calibration_samples * len(blank_inputs) * 3 # correct, incorrect, zeroshot
        for i in range(n_calibration_samples):
            for blank in blank_inputs:
                c, i, z = get_all_prompts_single_question(n_examples_per_class, text, labels, blank, dataset_bad_labels[dataset], dataset)
                for prompt in [c, i, z]:
                    calibration_inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
                    all_logits = get_first_token_logits_per_layer(model, calibration_inputs)
                    
                    for exit_layer in range(n_early_exit):
                        W_arrays[exit_layer] = np.add(W_arrays[exit_layer], compute_single_weight(all_logits[exit_layer], token_map))


        # For each exit layer, average over all weights and save out the weights
        print('Saving calibration matrices')
        for exit_layer in range(n_early_exit):
            path = './calibration/n_demos_' + str(n_demos) + '/' + dataset + '/' + model_name + '/exit_'  + str(exit_layer) + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            W_all = W_arrays[exit_layer] / total_calibration_prompts
            np.save(path + 'weights.npy', W_all)

        # Free up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
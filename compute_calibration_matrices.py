import pandas as pd
from utils import compute_single_weight, get_last_token_logits_per_layer
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
with open('dataset_labels.json') as f:
    dataset_labels = json.load(f)

# Get bad labels for generating incorrect examples
with open('dataset_bad_labels.json') as f:
    dataset_bad_labels = json.load(f)

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

with open('fake_labels.json') as f:
    fake_labels = json.load(f)

use_fake_labels = os.getenv('USE_FAKE_LABELS') == 'Y'

# datasets = ['ag_news', 'mqp', 'mrp', 'wnli', 'financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 
#             'unnatural', 'boolean', 'navigation', 'sports', 'web_of_lies']
# No too-hard datasets
datasets = ['trec', 'ag_news', 'financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 'unnatural']
dataset_idx = os.getenv('DATASET_INDEX')
if dataset_idx != 'a':
    # Just one dataset
    datasets = [datasets[int(dataset_idx)]]

for dataset in datasets:
    print(dataset, n_demos, 'demos')
    df = pd.read_csv('./processed_data/icl/' + dataset + '.csv')
    # Construct calibration prompts on the fly
    labels, text = df['label'], df['text']
    ds_labels, ds_bad_labels = dataset_labels[dataset], dataset_bad_labels[dataset]
    # Apply fake labels, if using
    if use_fake_labels:
        ds_labels, ds_bad_labels = [], {}
        for label in dataset_labels[dataset]:
            ds_labels.append(fake_labels[label])

        for key, value in dataset_bad_labels[dataset].items():
            ds_bad_labels[fake_labels[key]] = fake_labels[value]

        labels = [fake_labels[x] for x in labels]
    # Generate x=n_calibration_samples * num blank inputs (3) * n_classes (3) randomized examples
    n_calibration_samples = 250
    # Generate prompts on the fly
    n_examples_per_class = int(n_demos/len(ds_labels))

    for model_name, tokenizer_name, n_early_exit in zip(all_models, all_tokenizers, all_early_exits):
        print('Running model: ', model_name)
        token_map = token_maps[tokenizer_name]
        # Reduce down to only the allowed labels
        keys_to_remove = []
        for key in token_map:
            if key not in ds_labels:
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
                zeroshot_labels = dataset_labels[dataset] # specify the possible labels for zeroshot
                c, i, z = get_all_prompts_single_question(n_examples_per_class, text, labels, blank, ds_bad_labels, dataset, use_fake_labels)
                for prompt in [c, i, z]:
                    calibration_inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
                    all_logits = get_last_token_logits_per_layer(model, calibration_inputs)
                    
                    for exit_layer in range(n_early_exit):
                        W_arrays[exit_layer] = np.add(W_arrays[exit_layer], compute_single_weight(all_logits[exit_layer], token_map))


        # For each exit layer, average over all weights and save out the weights
        print('Saving calibration matrices')
        for exit_layer in range(n_early_exit):
            path = './calibration' + ('_fake_labels' if use_fake_labels else '') + '/n_demos_' + str(n_demos) + '/' + dataset 
            path += '/' + model_name + '/exit_'  + str(exit_layer) + '/'
            if not os.path.exists(path): 
                os.makedirs(path)

            W_all = W_arrays[exit_layer] / total_calibration_prompts
            np.save(path + 'weights.npy', W_all)

        # Free up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
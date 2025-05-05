# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from utils import get_intermediate_output_single_prompt
from prompt_utils import get_all_prompts_single_question
import pandas as pd
import numpy as np
import datetime

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

experiment_type = os.getenv('EXPERIMENT_TYPE')
n_demos = int(os.getenv('N_DEMOS'))
result_folder_name = os.getenv('RESULT_FOLDER_NAME')
use_calibration = os.getenv('USE_CALIBRATION') == 'Y'

# datasets = ['ag_news', 'mqp', 'mrp', 'wnli', 'financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 
#             'unnatural', 'boolean', 'navigation', 'sports', 'web_of_lies']
# No too-hard datasets
datasets = ['trec', 'ag_news', 'financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 'unnatural']
dataset_idx = int(os.getenv('DATASET_INDEX'))
dataset = datasets[dataset_idx]

# Additional models: "facebook/layerskip-llama3.2-1B", "meta-llama/Llama-3.2-1B"
models = ["facebook/layerskip-llama3-8B", "facebook/layerskip-llama2-7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
          'meta-llama/Llama-2-13B-hf']

tokenizers = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
              'meta-llama/Llama-2-13B-hf']

# Get the appropriate list of labels
with open('dataset_labels.json') as f:
    dataset_labels = json.load(f)

# Get bad labels for generating incorrect examples
with open('dataset_bad_labels.json') as f:
    dataset_bad_labels = json.load(f)

with open('fake_labels.json') as f:
    fake_labels = json.load(f)

use_fake_labels = os.getenv('USE_FAKE_LABELS') == 'Y'

# Load the dataset
data = pd.read_csv('./processed_data/icl/' + dataset + '.csv')
labels, text = data['label'], data['text']
ds_labels, ds_bad_labels = dataset_labels[dataset], dataset_bad_labels[dataset]
# Apply fake labels, if using
if use_fake_labels:
    ds_labels, ds_bad_labels = [], {}
    for label in dataset_labels[dataset]:
        ds_labels.append(fake_labels[label])

    for key, value in dataset_bad_labels[dataset].items():
        ds_bad_labels[fake_labels[key]] = fake_labels[value]

    labels = [fake_labels[x] for x in labels]
# Generate prompts on the fly
n_examples_per_class = int(n_demos/len(ds_labels))
correct_prompts, incorrect_prompts, zeroshot_prompts = [], [], []
for question in text.to_list():
    zeroshot_labels = dataset_labels[dataset] # specify the possible labels for zeroshot
    c, i, z = get_all_prompts_single_question(n_examples_per_class, text, labels, question, ds_bad_labels, dataset, use_fake_labels)
    correct_prompts.append(c)
    incorrect_prompts.append(i)
    zeroshot_prompts.append(z)

# Determine which model(s) to use
model_idx = os.getenv('MODEL_INDEX')
all_models, all_tokenizers = [], []
if model_idx == 'a':
    # Do all models!
    all_models, all_tokenizers = models, tokenizers
else:
    all_models, all_tokenizers = [models[int(model_idx)]], [tokenizers[int(model_idx)]]

for model_name, tokenizer_name in zip(all_models, all_tokenizers):
    print(model_name, dataset, n_demos, 'demos')

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)  
    print('Model loaded successfully. Device: ', next(model.parameters()).device)

    # Release memory from previous model
    torch.cuda.empty_cache()

    # Get the token_map of all possible token IDs corresponding to each of these classes. 
    with open('all_token_maps.json') as f:
        token_maps = json.load(f)
    token_map = token_maps[tokenizer_name]

    # Reduce down to only the allowed labels
    keys_to_remove = []
    for key in token_map:
        if key not in ds_labels:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        token_map.pop(key, None)
    
    # Define the filename for the calibration matrices to load from
    W_filepath = './calibration' + ('_fake_labels' if use_fake_labels else '') +'/n_demos_' + str(n_demos) + '/' + dataset + '/' 
    W_filepath += model_name + '/'

    # Break into cases by experiment type
    all_prompts, all_labels = [], []
    if experiment_type == 'a':
        all_prompts, all_labels = [correct_prompts, incorrect_prompts, zeroshot_prompts], ['correct', 'incorrect', 'zeroshot']
    elif experiment_type == 'c':
        all_prompts, all_labels = [correct_prompts], ['correct']
    elif experiment_type == 'z':
        all_prompts, all_labels = [zeroshot_prompts], ['zeroshot']
    elif experiment_type == 'i':
        all_prompts, all_labels = [incorrect_prompts], ['incorrect']

    # Run all the experiments for this model and dataset
    all_data = {}
    for prompts, expt_type in zip(all_prompts, all_labels):
        print('Running', expt_type)

        # Get the intermediate predictions from the model at each layer, for each question
        for i in range(len(prompts)):
            # get raw prediction results
            if use_calibration:
                results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, W_filepath)
            else:
                results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, None)

            results['true_label'] = str(labels[i]).lower()
            if len(all_data) == 0:
                # Initialize all_data
                for col in results:
                    all_data[col] = [results[col]]
            else:
                # Append to the list
                for col in results:
                    all_data[col].append(results[col])

        # Save result
        path = './' + result_folder_name + ('_fake_labels' if use_fake_labels else '') + ('/calibrated' if use_calibration else '/uncalibrated')
        path += '/n_demos_' + str(n_demos) + '/' + dataset + '/' + model_name + '/' 
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + expt_type + '.json', 'w') as file:
            json.dump(all_data, file)
        
        # Reset all_data for the next experiment
        all_data = {}
            
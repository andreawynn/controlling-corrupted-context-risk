# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from scratch.utils_backup import get_intermediate_output_single_prompt, save_single_result
import pandas as pd
import numpy as np

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

experiment_type = os.getenv('EXPERIMENT_TYPE')
n_demos = os.getenv('N_DEMOS')
result_folder_name = os.getenv('RESULT_FOLDER_NAME')

# Get the dataset name
datasets = ['financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 'unnatural']

# Additional models: "facebook/layerskip-llama3.2-1B", "meta-llama/Llama-3.2-1B"
models = ["facebook/layerskip-llama3-8B", "facebook/layerskip-llama2-7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
          'meta-llama/Llama-2-13B-hf', "meta-llama/Meta-Llama-3-70B"]

tokenizers = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
              'meta-llama/Llama-2-13B-hf', "meta-llama/Meta-Llama-3-70B"]

correct_filename, incorrect_filename, zeroshot_filename = 'correct.csv', 'incorrect.csv', 'zeroshot.csv'

dataset_idx = int(os.getenv('DATASET_INDEX'))
dataset = datasets[dataset_idx]

# Get the appropriate list of labels
dataset_labels = {
    'financial_phrasebank': ['positive', 'negative', 'neutral'],
    'sst2': ['positive', 'negative'],
    'tweeteval_hate': ['favor', 'against'],
    'tweeteval_atheism': ['yes', 'no', 'neither'],
    'tweeteval_feminist': ['yes', 'no', 'neither'],
    'unnatural': ['plant/vegetable', 'sport', 'animal'],
}

# Load the dataset
data = pd.read_csv('./datasets/' + dataset + '/' + n_demos + '_demos_balanced.csv')
labels = data['label']
correct_prompts, incorrect_prompts, zeroshot_prompts = data['correct_prompt'], data['incorrect_prompt'], data['zeroshot_prompt']

# Determine which model(s) to use
model_idx = os.getenv('MODEL_INDEX')
all_models, all_tokenizers = [], []
if model_idx == 'a':
    # Do all models!
    all_models, all_tokenizers = models, tokenizers
else:
    all_models, all_tokenizers = [models[int(model_idx)]], [tokenizers[int(model_idx)]]

for model_name, tokenizer_name in zip(all_models, all_tokenizers):
    print(model_name, dataset, experiment_type)
    
    # Define the filename for the calibration matrices to load from
    W_filepath = './calibration/' + dataset + '/' + model_name + '/'

    path = './' + result_folder_name + '/' + dataset + '/' + n_demos + '/' + model_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Release memory from previous model
    torch.cuda.empty_cache()

    # Break into cases by experiment type
    all_prompts, all_results_filenames = [], []

    if experiment_type == 'a':
        # Add all of them!
        all_prompts = [correct_prompts, incorrect_prompts, zeroshot_prompts]
        all_results_filenames = [correct_filename, incorrect_filename, zeroshot_filename]
    elif experiment_type == 'c':
        all_prompts, all_results_filenames = [correct_prompts], [correct_filename]
    elif experiment_type == 'z':
        all_prompts, all_results_filenames = [zeroshot_prompts], [zeroshot_filename]
    elif experiment_type == 'i':
        all_prompts, all_results_filenames = [incorrect_prompts], [incorrect_filename]

    # Get the token_map of all possible token IDs corresponding to each of these classes. 
    with open('all_token_maps_one_id.json') as f:
        token_maps = json.load(f)
    token_map = token_maps[tokenizer_name]

    # Reduce down to only the allowed labels
    keys_to_remove = []
    for key in token_map:
        if key not in dataset_labels[dataset]:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        token_map.pop(key, None)

    # Run all the experiments for this model and dataset
    for prompts, results_filename in zip(all_prompts, all_results_filenames):
        # Check for the starting point
        start_index = 0
        if os.path.exists(path + results_filename):
            df = pd.read_csv(path + results_filename, engine='python', on_bad_lines='warn').dropna()
            start_index = df.shape[0]

        # Get the intermediate predictions from the model at each layer
        for i in range(start_index, data.shape[0]):
            if i % 100 == 0:
                print(i)

            torch.cuda.empty_cache()
            # get raw prediction results
            results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, 1, W_filepath)
            # no calibration
            # results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, 1)
            # get full model generation
            # full_generation_output = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, None, 1, W_filepath)
            # # combine the two
            # for exit_layer in range(1, len(model.model.layers)):
            #     results[str(exit_layer) + '_generation'] = full_generation_output[str(exit_layer)]
            #     results['max_logit' + str(exit_layer)] = full_generation_output['max_logit' + str(exit_layer)]
            #     results['max_token_id' + str(exit_layer)] = full_generation_output['max_token_id' + str(exit_layer)]
            # results['full_model_generation'] = full_generation_output['full_model']

            results['true_label'] = labels[i]
            save_single_result(results, path, results_filename)
            

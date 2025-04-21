# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from utils import get_intermediate_output_single_prompt, save_single_result
from prompt_utils import get_all_prompts_single_question
import pandas as pd
import numpy as np
import datetime

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

experiment_type = os.getenv('EXPERIMENT_TYPE')
n_demos = int(os.getenv('N_DEMOS'))
result_folder_name = os.getenv('RESULT_FOLDER_NAME')

# Get the dataset name
datasets = ['financial_phrasebank', 'sst2', 'tweeteval_hate', 'tweeteval_feminist', 'tweeteval_atheism', 'unnatural',
            'boolean', 'navigation', 'sports', 'web_of_lies']
dataset_idx = int(os.getenv('DATASET_INDEX'))
dataset = datasets[dataset_idx]

# Additional models: "facebook/layerskip-llama3.2-1B", "meta-llama/Llama-3.2-1B"
models = ["facebook/layerskip-llama3-8B", "facebook/layerskip-llama2-7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
          'meta-llama/Llama-2-13B-hf']

tokenizers = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
              'meta-llama/Llama-2-13B-hf']

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

# Load the dataset
data = pd.read_csv('./datasets/' + dataset + '/risk_control_data.csv')
labels, text = data['label'], data['text']
# Generate prompts on the fly
n_examples_per_class = int(n_demos/len(dataset_labels[dataset]))
correct_prompts, incorrect_prompts, zeroshot_prompts = [], [], []
for question in text.to_list():
    c, i, z = get_all_prompts_single_question(n_examples_per_class, text, labels, question, dataset_bad_labels[dataset], dataset)
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
    print(model_name, dataset)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print('Tokenizer loaded, loading model')
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
        if key not in dataset_labels[dataset]:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        token_map.pop(key, None)
    
    # Define the filename for the calibration matrices to load from
    W_filepath = './calibration/n_demos_' + str(n_demos) + '/' + dataset + '/' + model_name + '/'

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
    all_data = None
    for prompts, expt_type in zip(all_prompts, all_labels):
        print('Running', expt_type)
        path = './' + result_folder_name + '/' + dataset + '/n_demos_' + str(n_demos) + '/' + model_name + '/' + expt_type + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Get the intermediate predictions from the model at each layer, for each question
        for i in range(data.shape[0]):
            torch.cuda.empty_cache()
            # get raw prediction results
            results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, W_filepath)
            results['true_label'] = labels[i]
            if all_data is None:
                # Initialize all_data
                all_data = {}
                for col in results:
                    all_data[col] = [results[col]]
            else:
                # Append to the list
                for col in results:
                    all_data[col].append(results[col])
            
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

            # save_single_result(results, path, datetime_string + '.csv')

    # Save results to CSV
    now = datetime.datetime.now()
    datetime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame()
    for col in all_data:
        df[col] = all_data[col]
    df.to_csv(path + datetime_string + '.csv', index=False)
            

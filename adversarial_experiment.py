# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import numpy as np
from adversarial_utils import llama_guard_safety_score
from utils import generate_sequence_and_probs
import datetime

from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

result_folder_name = os.getenv('RESULT_FOLDER_NAME')
early_exit = os.getenv('EARLY_EXIT') == "Y"

# Get the dataset name
datasets = ['alert']
dataset_idx = int(os.getenv('DATASET_INDEX'))
dataset = datasets[dataset_idx]

num_candidate_responses = int(os.getenv('NUM_CANDIDATE_RESPONSES'))

# Additional models: "facebook/layerskip-llama3.2-1B", "meta-llama/Llama-3.2-1B"
models = ["facebook/layerskip-llama3-8B", "facebook/layerskip-llama2-7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
          'meta-llama/Llama-2-13B-hf']

tokenizers = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7B-hf", 
              'meta-llama/Llama-2-13B-hf']

# Load the prompts
use_adv_prompts = os.getenv("USE_ADVERSARIAL_PROMPTS") == "Y"
filename = '/adv_prompts.json' if use_adv_prompts else '/prompts.json'
with open('./datasets/adversarial/' + dataset + filename, 'r') as file:
    data = json.load(file)
prompts = data['prompt']

# Load the guard model (so we only do this once)
guard_model_name = "meta-llama/LlamaGuard-7b"
guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_name)
guard_model = AutoModelForCausalLM.from_pretrained(guard_model_name, torch_dtype=torch.bfloat16, device_map="auto")

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
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)  
    print('Model loaded successfully. Device: ', next(model.parameters()).device)

    # Release memory from previous model
    torch.cuda.empty_cache()

    # Run all the experiments for this model and dataset
    all_data = {'prompt': prompts, 'responses': [], 'risks': [], 'probs': []}
    now = datetime.datetime.now()
    datetime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    for prompt in prompts:
        if early_exit:
            print('Not implemented')
            # results = get_intermediate_output_single_prompt(prompts[i], model, tokenizer, token_map, None)
            # results['true_label'] = str(labels[i]).lower()
            # if len(all_data) == 0:
            #     # Initialize all_data
            #     for col in results:
            #         all_data[col] = [results[col]]
            # else:
            #     # Append to the list
            #     for col in results:
            #         all_data[col].append(results[col])
        else:
            # Generate num_candidate_responses responses for each question (perhaps varying temperature of model?)
            inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            responses, risks, probs = [], [], []
            for i in range(num_candidate_responses):
                response, all_probs = generate_sequence_and_probs(model, tokenizer, inputs, max_tokens=100, temperature=0.1)
                # Get the measure of risk for each response
                risks.append(llama_guard_safety_score(prompt, response, guard_model, guard_tokenizer))
                responses.append(response)
                probs.append(all_probs)

        # Add to all_data
        all_data['responses'].append(responses)
        all_data['risks'].append(risks)
        all_data['probs'].append(probs)

    # Save result
    path = './' + result_folder_name + '/' + dataset + ('_adv' if use_adv_prompts else '') + '/' + model_name + '/' 
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + datetime_string + '.json', 'w') as file:
        json.dump(all_data, file)
            

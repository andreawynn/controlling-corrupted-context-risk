from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import requests
import os
import random
import numpy as np
import pandas as pd
import transformers
from typing import List, Optional, Tuple


# Compute phat given a set of logits and token map
def compute_phat(logits, token_map):
    # Step 1: Find the logits we care about
    phat_cf = get_max_class(token_map, logits, return_all_logits=True)
    
    # Step 2: Re-normalize to 1 (i.e. make them valid probabilities)
    # logit_sum = sum(list(phat_cf.values()))
    # for label in phat_cf:
    #     phat_cf[label] = phat_cf[label] / logit_sum

    # Apply softmax
    phat_cf_list = []
    label_order = list(phat_cf.keys())
    for label in phat_cf:
        phat_cf_list.append(phat_cf[label])
    phat_cf = torch.nn.functional.softmax(torch.tensor(phat_cf_list), dim=0)

    # Convert back to dict format
    phat_cf_dict = {}
    for i in range(len(label_order)):
        phat_cf_dict[label_order[i]] = phat_cf[i]

    return phat_cf_dict


def compute_single_weight(calibration_logits, token_map):
    # Pre-define an ordering of the labels such that their use can be consistent across all matrix operations
    label_order = list(token_map.keys())
    
    # Step 1-2: compute normalized calibration logits
    phat_cf = compute_phat(calibration_logits, token_map)

    # Step 3: Set weight matrix W = inverse(diag(phat_cf)) and biases b=0. 
    W = []
    for idx in range(len(label_order)):
        label = label_order[idx]
        row = np.zeros(len(label_order))
        row[idx] = 1 / phat_cf[label]
        W.append(row)

    return W


# Loading in a pre-computed calibration matrix
def compute_calibrated_prediction(logits, W, token_map, return_logit_value=False):
    label_order = list(token_map.keys())

    # Compute phat for the logits (as a vector in the correct order)
    phat = compute_phat(logits, token_map)
    phat_vec = []
    for label in label_order:
        phat_vec.append([phat[label]])
    phat_vec = np.array(phat_vec)

    # Step 4: For new input, compute W * phat + b = W * phat (since b=0)
    qhat = np.matmul(W, phat_vec)

    # Get the max class prediction and return it
    max_class_idx = np.argmax(qhat.flatten())
    if return_logit_value:
        return qhat.flatten(), label_order, label_order[max_class_idx]
    
    return label_order[max_class_idx]


# Using the maximum logit for each label
def get_max_class(token_map, token_logits, return_all_logits=False):
    last_token_logits = token_logits[0,:]
    output_logits = {}
    for l in token_map:
        # For all the token IDs corresponding to this label, add the logits. 
        max_label_logit = last_token_logits[token_map[l][0]].item()
        for token_id in token_map[l]:
            max_label_logit = max(max_label_logit, last_token_logits[token_id].item())
        output_logits[l] = max_label_logit

    if return_all_logits:
        return output_logits
    # Get the token (within the valid options) that has the maximum logit value
    return max(output_logits, key=output_logits.get)
    

# output_classes determines whether output should be restricted to a single list of single-token outputs
# if None, do not restrict output. otherwise, it should be a list containing all the possible outputs. 
# e.g. true/false, positive/negative/neutral, ...
def get_intermediate_output_single_prompt(prompt, model, tokenizer, token_map, calibration_weight_filepath=None): 
    single_prompt_results = {'prompt': prompt}
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_attention_mask=True, return_tensors="pt")

    # Decode to a restricted set of classes specified by token_map. 
    # In this case, we can be more efficient by pre-computing all early exits at once.
    all_logits = get_first_token_logits_per_layer(model, inputs)
        
    # Compute outputs at each intermediate layer
    for exit_layer in range(len(model.model.layers)):
        # Load in the appropriate calibration matrix from the folder
        W = None
        if calibration_weight_filepath is not None:
            # Take moving average over 3 exits
            W = np.load(calibration_weight_filepath + 'exit_'  + str(exit_layer) + '/weights.npy')
            if exit_layer == 0 or exit_layer == len(model.model.layers)-1:
                # final exit - moving avg with the last three
                W = np.load(calibration_weight_filepath + 'exit_0/weights.npy')
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_'  + str(len(model.model.layers)-1) + '/weights.npy'))
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_'  + str(len(model.model.layers)-2) + '/weights.npy'))
            elif exit_layer == 1:
                # first exit - moving avg with the first three
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_2/weights.npy'))
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_3/weights.npy'))
            else:
                # moving avg with one before, one after
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_'  + str(exit_layer-1) + '/weights.npy'))
                W = np.add(W, np.load(calibration_weight_filepath + 'exit_'  + str(exit_layer+1) + '/weights.npy'))
            # Divide by 3 to get avg
            W = W / 3

        # Make prediction with contextual calibration on a single token. 
        # This is where filtering down to the valid token IDs is done (in the functions)
        logits = all_logits[len(model.model.layers)-1 if exit_layer == 0 else exit_layer-1]
        if W is not None:
            token_logits, label_order, intermediate_output = compute_calibrated_prediction(logits, W, token_map, return_logit_value=True)
            # Save out the confidence score of the max logit
            confidences = torch.nn.functional.softmax(torch.tensor(token_logits), dim=0)
            # Only use code below if I want to save the individual logits (for debugging/examination)
            # for i in range(len(label_order)):
            #     single_prompt_results[str(exit_layer) + label_order[i] + '_confidence'] = token_logits[i]
        else:
            intermediate_output = get_max_class(token_map, logits)
        
        col = 'full_model' if exit_layer == 0 else str(exit_layer)
        single_prompt_results[col] = intermediate_output
        single_prompt_results[col + '_confidence'] = max(confidences).item()
        single_prompt_results[col + '_raw_logits'] = token_logits

    return single_prompt_results

def save_single_result(single_prompt_results, path, filename):
    # Create the columns
    columns = list(single_prompt_results.keys()) 
    
    # Put each element in a list so it can be parsed to a DataFrame row
    for c in columns:
        single_prompt_results[c] = [single_prompt_results[c]]
    single_prompt = pd.DataFrame(single_prompt_results, columns=columns)
    
    # Save the data
    if os.path.exists(path + filename):
        results = pd.concat([pd.read_csv(path + filename), single_prompt])
    else:
        results = single_prompt
    results.to_csv(path + filename, index=False)


def get_first_token_logits_per_layer(model, inputs):
    """
    Given a HuggingFace AutoModelForCausalLM and tokenized inputs, 
    return the logits for the first token output from each layer of the model.

    Args:
        model (AutoModelForCausalLM): The pre-trained language model.
        inputs (dict): Tokenized inputs from the tokenizer (e.g., output of tokenizer()).

    Returns:
        list: A list of tensors, where each tensor contains the logits for the first token
              from each layer of the model.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # (/home/awynn/scratchenalisn1/awynn/anaconda3/envs/llm-risk-control)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)  

    # Extract hidden states from all layers
    hidden_states = outputs.hidden_states  # Tuple of tensors, one for each layer

    # Extract logits for the first token from each layer
    logits_per_layer = []
    for layer_idx, layer_hidden_states in enumerate(hidden_states):
        # Take the first token's hidden state (index 0 in sequence dimension)
        first_token_hidden_state = layer_hidden_states[:, 0, :]  # Shape: [batch_size, hidden_size]
        # Project the hidden state to logits using the model's language modeling head
        logits = model.lm_head(first_token_hidden_state)  # Shape: [batch_size, vocab_size]
        logits_per_layer.append(logits)

    return logits_per_layer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt


def get_intermediate_embeddings(model, inputs, max_length):
    # Retrieve outputs from intermediate layers using a hook
    # Dictionary to store intermediate outputs
    intermediate_outputs = {}
    
    # Define a hook function to capture the output of a specific layer
    def hook_fn(module, input, output):
        # Find the layer name
        for name, curr_module in model.named_modules():
            if curr_module is module:
                layer_name = name
                
        if layer_name not in intermediate_outputs:
            intermediate_outputs[layer_name] = []
        intermediate_outputs[layer_name].append(output)
    
    # Note: Adjust the layers you want to hook based on the model's architecture
    hook_handles = []
    for idx, layer in enumerate(model.model.layers):
        # Hook the output of the attention mechanism (self-attention)
        hook_handles.append(layer.self_attn.o_proj.register_forward_hook(hook_fn))
        # Hook the last layer of the feed forward network
        hook_handles.append(layer.post_attention_layernorm.register_forward_hook(hook_fn))
    
    # Perform a forward pass on the already-tokenized sample prompt
    with torch.no_grad():
        out = model.generate(**inputs, num_return_sequences=1, max_length=max_length)

    # Remove all the hooks
    for handle in hook_handles:
        handle.remove()

    return intermediate_outputs


def get_intermediate_text_outputs(model, tokenizer, intermediate_outputs):
    unembedding_matrix = model.get_output_embeddings().weight
    intermediate_text = {}
    
    for layer_name, outputs in intermediate_outputs.items():
        # Decode one token at a time. 
        predicted_token_ids = []
        for k in range(len(outputs)):
            output_tensor = outputs[k].squeeze(0)
            # Apply the unembedding matrix
            logits = torch.matmul(output_tensor, unembedding_matrix.T)
            # Note: the model still produces sequence lengths of 20 in intermediate generations.
            # Select just the first one. 
            next_token_id = torch.argmax(logits, dim=-1)[0]
            # Break if the token is EOS
            if next_token_id == tokenizer.eos_token_id:
                break
            # Otherwise append to the predictions
            predicted_token_ids.append(next_token_id)
                            
        predicted_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        intermediate_text[layer_name] = predicted_text

    return intermediate_text


# Get scores according to some specific metric using the given model. 
# 
# model: the language model to run
# get_prompt: given index i in the dataset, return the ith prompt
# limit: max iterations to run. Set limit to len(dataset) to go through the full dataset.
# print_every: how often to print a progress tracker
# filepath: the FOLDER (not full filename) for saving out data. should include model name!
# score_fn: a function that, given a single string of text, scores the text
def get_all_intermediate_text_dataset(model, tokenizer, get_prompt, limit, print_every=100): 
    # Format: each layer maps to a list of the outputs per prompt. 
    all_intermediate_outputs = {}
    for i in range(limit):
        # Progress tracker
        if i % print_every == 0:
            print(i, 'out of', limit)
        
        # Tokenize the prompt
        inputs = tokenizer(get_prompt(i), return_tensors="pt").to(device)
        intermediate_embeddings = get_intermediate_embeddings(model, inputs, 50)
        intermediate_text = get_intermediate_text_outputs(model, tokenizer, intermediate_embeddings)
        
        for layer_name, text in intermediate_text.items():
            if layer_name not in all_intermediate_outputs:
                all_intermediate_outputs[layer_name] = []
            all_intermediate_outputs[layer_name].append(text)

    return all_intermediate_outputs


def get_scores(all_intermediate_outputs, score_fn, filepath=None):
    # Map each layer to a list of scores for each prompt. 
    # Score has to be per prompt; e.g. toxicity of prompt, correctness, etc. 
    # This is important since we later will want to do things like manage risk over the entire population. 
    scores = { 'o_proj': {}, 'post_attention_layernorm': {}}
    
    # Get the score using the provided function for the text output from each layer. 
    # If classification/accuracy, score is 0 or 1 (or appropriate equivalent) for the prompt. 
    for layer_name, text in intermediate_text.items():
        score = score_fn(text)
        if 'o_proj' in layer_name:
            if layer_name not in scores['o_proj']:
                scores['o_proj'][layer_name] = []
            scores['o_proj'][layer_name].append(score)
        else:
            if layer_name not in scores['post_attention_layernorm']:
                scores['post_attention_layernorm'][layer_name] = []
            scores['post_attention_layernorm'][layer_name].append(score)
    
    # Save out the scores in a file
    if filepath is not None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = filepath + 'limit_' + str(limit) + '.json'
        with open(filename, 'w') as file:
            json.dump(scores, file, default=lambda o: float(o) if isinstance(o, np.float32) else o)

    return scores


def get_toxicity_prompt(i):
    return toxicity_prompts_dataset[i]['prompt']['text']
    

def detoxify_score(text):
    detoxify_model = Detoxify('original')
    return detoxify_model.predict(text)['toxicity']


def perspective_score(text):
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}}
    }
    response = requests.post(url, params={"key": api_key}, json=data)
    return response.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]




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
        out = model.generate(inputs['input_ids'], num_return_sequences=1, max_length=max_length)

    # Remove all the hooks
    for handle in hook_handles:
        handle.remove()

    return intermediate_outputs


def get_intermediate_text_outputs(model, tokenizer, intermediate_outputs, max_sequence_length):
    unembedding_matrix = model.get_output_embeddings().weight
    intermediate_text = {}
    
    for layer_name, outputs in intermediate_outputs.items():
        # Decode one token at a time. 
        predicted_token_ids = []
        for k in range(max_sequence_length):
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




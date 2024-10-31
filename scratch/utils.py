from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt
import os
import numpy as np
import transformers


# Based on the implementation from: https://github.com/facebookresearch/LayerSkip/blob/main/self_speculation/llama_model_utils.py#L213
def get_next_logits(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    exit_layer: int,
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    seq_length_with_past = seq_length
    past_key_values_length = 0
    past_key_values = None

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    
    # Constructing an attention mask of the same shape as the input IDs. 
    attention_mask = input_ids.new_ones( 
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    # Propagate through the decoder layers up until the early exit layer
    for decoder_layer in model.model.layers[:exit_layer]:
        if past_key_values is not None:
            hidden_states, past_key_values = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,)
        else:
            hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,)

    # past_key_values = past_key_values.to_legacy_cache()

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    return logits


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

# Decoding via sampling, not just argmax
def decode_next_token(
    logits: torch.Tensor,
    sample: bool = False,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        next_token.transpose_(1, 0)
        return next_token, probabilities


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)




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
def get_scores(model, tokenizer, device, get_prompt, score_fn, limit, filepath=None, print_every=100): 
    # Map each layer to a list of scores for each prompt. 
    # Score has to be per prompt; e.g. toxicity of prompt, correctness, etc. 
    # This is important since we later will want to do things like manage risk over the entire population. 
    scores = { 'o_proj': {}, 'post_attention_layernorm': {}}

    if filepath is not None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    # Format: each layer maps to a list of the outputs per prompt. 
    for i in range(limit):
        # Progress tracker
        if i % print_every == 0:
            print(i, 'out of', limit)
        
        # Tokenize the prompt
        inputs = tokenizer(get_prompt(i), return_tensors="pt").to(device)
        intermediate_embeddings = get_intermediate_embeddings(model, inputs, 50)
        intermediate_text = get_intermediate_text_outputs(model, tokenizer, intermediate_embeddings)
        
        # Get scores for all intermediate outputs for this iteration
        for layer_name, text in intermediate_text.items():
            # Get the score using the provided function for the text output from each layer. 
            # If classification/accuracy, score is 0 or 1 (or appropriate equivalent) for the prompt. 
            score = score_fn(text)
            if 'o_proj' in layer_name:
                if layer_name not in scores['o_proj']:
                    scores['o_proj'][layer_name] = []
                scores['o_proj'][layer_name].append(score)
            else:
                if layer_name not in scores['post_attention_layernorm']:
                    scores['post_attention_layernorm'][layer_name] = []
                scores['post_attention_layernorm'][layer_name].append(score)
    
        # Save out the scores in a file (do this every iteration!)
        if filepath is not None:
            filename = filepath + 'limit_' + str(limit) + '.json'
            with open(filename, 'w') as file:
                json.dump(scores, file, default=lambda o: float(o) if isinstance(o, np.float32) else o)

    return scores
    

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




from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import transformers
from typing import List, Optional, Tuple


def get_nq_open_fewshot_prompt(question_text, example_texts, example_answers):
    # Note: we can just use a simple fixed prompt. Examples taken from training set. 
    prompt = "Your job is to answer trivia questions correctly. You will be given a single question and asked to produce a single response. "
    prompt += "Here are a few examples of question-answer pairs for previous trivia questions. Your answer should follow the same format. \n"
    for i in range(len(example_texts)):
        prompt += "- Question " + str(i+1) + ": " + example_texts[i] + " || Answer " + str(i+1) + ": " + example_answers[i] + "\n"
    prompt += "Here is the question you should answer. Question " + str(len(example_texts)+1) + ": " + question_text 
    prompt += " || Answer " + str(len(example_texts)+1) + ": "
    return prompt


def get_financial_phrasebank_prompt(question_text, example_texts, example_answers):
    # Assumes example_texts and example_answers are the same length. 
    prompt = "Your job is to classify the sentiment of a given snippet of text. The possible classes are: positive, negative, neutral. "
    prompt += "Output only the class of the text snippet and nothing else. Below are a few examples of text-sentiment pairs. "
    prompt += "Your answer should follow the same format. \n"
    for i in range(len(example_texts)):
        prompt += "- Text " + str(i+1) + ": " + example_texts[i] + " || Answer " + str(i+1) + ": " + example_answers[i] + "\n"
    prompt += "Here is the text you should classify. Text " + str(len(example_texts)+1) + ": " + question_text 
    prompt += " || Answer " + str(len(example_texts)+1) + ": "
    return prompt


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


def get_max_class(options, tokenizer, output_ids):
    output_logits = {}
    for text in options:
        # Compute the token ID for this option (assumes it's just 1)
        token_ids = tokenizer(text)['input_ids'][1]
        # Get the model output logit for this specific token
        output_logits[text] = output_ids[0][0][token_ids].item()
    
    # Get the token (within the valid options) that has the maximum logit value
    return max(output_logits, key = output_logits.get)
    

# output_classes determines whether output should be restricted to a single list of single-token outputs
# if None, do not restrict output. otherwise, it should be a list containing all the possible outputs. 
# e.g. true/false, positive/negative/neutral, ...
def get_intermediate_output_single_prompt(prompt, path, filename, model, tokenizer, output_classes = None, max_new_tokens=50):
    single_prompt_results = {'prompt': prompt}
    columns = ['prompt', 'full_model']
    n_layers = len(model.model.layers)
    for i in range(1, n_layers): 
        columns.append(str(i))
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_attention_mask=True, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids'].clone().detach().to(model.device)
    
    # Compute outputs at each intermediate layer
    for exit_layer in range(len(model.model.layers)):
        # Either decode token-by-token, or decode restricted to a specific set of potential classes (output_classes)
        if output_classes is None:
            output_ids = decode_logits(model=model, input_ids=input_ids, max_new_tokens=max_new_tokens, exit_layer=exit_layer, eos_token_id=tokenizer.eos_token_id)
            intermediate_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            output_ids = decode_logits(model=model, input_ids=input_ids, max_new_tokens=max_new_tokens, exit_layer=exit_layer, return_all_logits=True, eos_token_id=tokenizer.eos_token_id)
            intermediate_output = get_max_class(output_classes, tokenizer, output_ids)
        
        if exit_layer == 0:
            single_prompt_results['full_model'] = intermediate_output
        else:
            single_prompt_results[str(exit_layer)] = intermediate_output
    
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


# Decode the logits according to an autoregressive strategy
# Based on the following: https://github.com/facebookresearch/LayerSkip/blob/main/self_speculation/autoregressive_generator.py#L45
def decode_logits(model, input_ids, max_new_tokens, exit_layer, eos_token_id, past_key_values=None, sample=True, return_all_logits=False,
                  temperature: Optional[float] = 0.7,
                  top_k: Optional[int] = 50,
                  top_p: Optional[float] = 0.95):
    output_ids = []
    exit_query_cache = None
    # Generate new tokens in a loop. 
    for _ in range(max_new_tokens):
        if exit_layer > 0:
            model_output = forward_early(
                model,
                input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
        else:
            model_output = forward(
                model,
                input_ids,
                past_key_values,
            )
        logits = model_output['logits']
        if return_all_logits:
            return logits
        past_key_values = model_output['past_key_values']
        next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
        next_token = next_token.item()
        if next_token == eos_token_id:
            break
        output_ids.append(next_token)
        # Don't concatenate `next_token` to original `input_ids` since we're using
        # the KV cache (`past_key_values`) to speed up generation.
        input_ids = torch.tensor([[next_token]]).to(input_ids)

    return output_ids


def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
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
    for decoder_layer in model.model.layers:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return {'logits': logits, 'past_key_values': past_key_values}


# Adapting intermediate decoding from here: https://github.com/facebookresearch/LayerSkip/blob/main/self_speculation/llama_model_utils.py#L213
# Returns {logits, past_key_values, exit_query_cache}
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
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
    for decoder_layer in model.model.layers[:exit_layer]:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    # Returns (logits, past_key_values, exit_query_cache)
    return {'logits': logits, 'past_key_values': past_key_values, 'exit_query_cache': exit_query_cache}
    

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
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        if not token_idx:
            logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
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
    




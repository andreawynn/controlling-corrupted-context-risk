from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from detoxify import Detoxify
import requests
import matplotlib.pyplot as plt




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

def get_toxicity_scores(intermediate_text, score_name, score_fn, scores):
    scores['o_proj'][score_name] = []
    scores['post_attention_layernorm'][score_name] = []
    
    # Get the score using the provided function for the text output from each layer. 
    for layer_name, text in intermediate_text.items():
        score = score_fn(text)
        if 'o_proj' in layer_name:
            scores['o_proj'][score_name].append(score)
        else:
            scores['post_attention_layernorm'][score_name].append(score)

    return scores


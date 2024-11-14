# ILPR
import json
import torch
from transformers import T5TokenizerFast

# Load tokenizer
tokenizer = T5TokenizerFast.from_pretrained('model_t5/t5-base')

# Loading change information
# with open('intervention_strategies.json', 'r') as f:
with open('game_intervention_strategies.json', 'r') as f:
    intervention_strategies = json.load(f)

# Vectorized change information
vectorized_strategies = {}
for key, value in intervention_strategies.items():
    tokenized_key = tokenizer(key, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    tokenized_value = tokenizer(value, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    vectorized_strategies[key] = {
        'input_ids': tokenized_key['input_ids'].tolist(),
        'attention_mask': tokenized_key['attention_mask'].tolist(),
        'labels': tokenized_value['input_ids'].tolist()
    }

# Save vectorized changes
with open('game_vectorized_intervention_strategies.json', 'w') as f:
    json.dump(vectorized_strategies, f)




# CLPR
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
#
# def calculate_cosine_similarity(vec1, vec2):
#     similarity = cosine_similarity(vec1, vec2)
#     return similarity
#
#
# vectorized_strategies = {}
# for key, value in intervention_strategies.items():
#     tokenized_key = tokenizer(key, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     tokenized_value = tokenizer(value, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#
#     input_ids_key = tokenized_key['input_ids'].squeeze(0)
#     input_ids_value = tokenized_value['input_ids'].squeeze(0)
#
#     similarity = calculate_cosine_similarity(input_ids_key.numpy().reshape(1, -1),
#                                              input_ids_value.numpy().reshape(1, -1))
#
#     if similarity > 0.8:
#         vectorized_strategies[key] = {
#             'input_ids': tokenized_key['input_ids'].tolist(),
#             'attention_mask': tokenized_key['attention_mask'].tolist(),
#             'labels': tokenized_value['input_ids'].tolist()
#         }
#
# with open('game_vectorized_intervention_strategies.json', 'w') as f:
#     json.dump(vectorized_strategies, f)

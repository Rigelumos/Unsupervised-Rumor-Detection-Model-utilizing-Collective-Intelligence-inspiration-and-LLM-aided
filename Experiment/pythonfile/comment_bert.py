import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def compute_cosine_similarity(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cluster_comments(comments, threshold=0.85):
    clusters = []
    cluster_labels = []

    for comment in comments:
        text_vec = get_embedding(comment)

        if not clusters:
            clusters.append(text_vec)
            cluster_labels.append(0)
            continue

        max_similarity = 0
        best_cluster = None
        for i, cluster_vec in enumerate(clusters):
            sim = compute_cosine_similarity(text_vec, cluster_vec)
            if sim > max_similarity:
                max_similarity = sim
                best_cluster = i

        if max_similarity >= threshold:
            cluster_labels.append(best_cluster)
        else:
            clusters.append(text_vec)
            cluster_labels.append(len(clusters) - 1)

    return cluster_labels

input_json = "input.json"
with open(input_json, encoding='utf-8') as file:
    json_data = json.load(file)

for item in json_data:
    comments = [comment['comment_text'] for comment in item['comments']]

    if not comments:
        continue

    cluster_labels = cluster_comments(comments, threshold=0.85)

    for idx, comment in enumerate(item['comments']):
        comment['cluster'] = cluster_labels[idx]

output_json = "output.json"
with open(output_json, 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)


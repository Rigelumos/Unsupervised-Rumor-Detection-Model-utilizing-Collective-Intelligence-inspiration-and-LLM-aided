from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer('sentence-transformers-model')

with open('fake_all.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    text = item.get("text", "")
    if text:
        text_embedding = model.encode(text).tolist()
        item["text_embedding"] = text_embedding

    comments = item.get("comments", [])
    sentiment_groups = {}

    for comment in comments:
        sentiment = comment.get("sentiment", "undefined")
        comment_text = comment.get("comment_text", "").strip()

        if not comment_text:
            continue

        if sentiment not in sentiment_groups:
            sentiment_groups[sentiment] = []

        sentiment_groups[sentiment].append(comment_text)

    sentiment_embeddings = {}
    for sentiment, texts in sentiment_groups.items():
        try:
            embeddings = model.encode(texts)
            sentiment_embeddings[sentiment] = embeddings.tolist()
            print(f"Sentiment group '{sentiment}' has embeddings with shape {embeddings.shape}.")
        except Exception as e:
            print(f"Error generating embeddings for sentiment group '{sentiment}': {e}")
            sentiment_embeddings[sentiment] = [[0] * 384]

    item["sentiment_embeddings"] = sentiment_embeddings

with open('fake_embedding.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

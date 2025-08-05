import json
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
import csv
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracies = []
precisions = []
f1s = []
recalls = []

def get_answer(df, id):
    beacon_doubt = np.zeros(384)
    beacon_neutral = np.zeros(384)
    beacon_support = np.zeros(384)

    for _, item in df.iterrows():
        if item['id'] == id:
            item['dis_label'] = item['label']
            beacon_text = item['text_embedding']
            beacon_attitude = item['attitude']
            if item['sentiment'] == "质疑":
                beacon_doubt = item['embedding']
            elif item['sentiment'] == "中立":
                beacon_neutral = item['embedding']
            elif item['sentiment'] == "支持":
                beacon_support = item['embedding']

    for index, row in df.iterrows():
        if row['id'] != id:
            text_distance = euclidean(beacon_text, row['text_embedding'])
            attitude_distance = euclidean(beacon_attitude, row['attitude'])
            if row['sentiment'] == "质疑":
                sentiment_distance = euclidean(beacon_doubt, row['embedding'])
            elif row['sentiment'] == "中立":
                sentiment_distance = euclidean(beacon_neutral, row['embedding'])
            elif row['sentiment'] == "支持":
                sentiment_distance = euclidean(beacon_support, row['embedding'])
            df.at[index, 'text_distance'] = text_distance
            df.at[index, 'attitude_distance'] = attitude_distance
            df.at[index, 'sentiment_distance'] = sentiment_distance

    nested_dict = {
        row["id"]: {
            "label": row["label"],
            "text": None,
            "attitude": None,
            "支持": None,
            "中立": None,
            "质疑": None
        }
        for _, row in df.iterrows()
    }

    for _, row in df.iterrows():
        id_val = row["id"]
        sentiment = row["sentiment"]
        if id_val in nested_dict:
            nested_dict[id_val]["text"] = row["text_distance"]
            nested_dict[id_val]["attitude"] = row["attitude_distance"]
            nested_dict[id_val][sentiment] = row["sentiment_distance"]

    nested_dict = {
        key: value
        for key, value in nested_dict.items()
        if not any(math.isnan(v) for v in value.values() if isinstance(v, float))
    }

    keys = []
    feature_vectors = []

    for key, values in nested_dict.items():
        keys.append(key)
        feature_vector = [values["text"], values["attitude"], values["支持"], values["中立"], values["质疑"]]
        feature_vector = [0 if v is None else v for v in feature_vector]
        feature_vectors.append(feature_vector)

    feature_vectors = np.array(feature_vectors)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(feature_vectors)

    for i, key in enumerate(keys):
        nested_dict[key]["cluster"] = 1 - int(labels[i])

    true_labels = [nested_dict[key]["label"] for key in keys]
    predicted_labels = [nested_dict[key]["cluster"] for key in keys]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    return true_labels, predicted_labels

with open("input.json", 'r', encoding="utf-8") as f:
    data = json.load(f)

rows = []
for entry in data:
    for sentiment, embedding in entry["sentiment_embeddings"].items():
        rows.append({
            "id": entry["ID"],
            "label": entry["label"],
            "attitude": entry["attitude"],
            "text_embedding": entry["text_embedding"],
            "sentiment": sentiment,
            "embedding": embedding
        })

df = pd.DataFrame(rows)

beacon_id = "***"

true_labels, predicted_labels = get_answer(df, beacon_id)
print(true_labels)
print(predicted_labels)

with open("output.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "accuracies", "precisions", "f1s", "recalls"])
    writer.writerow([beacon_id, accuracies[0], precisions[0], f1s[0], recalls[0]])


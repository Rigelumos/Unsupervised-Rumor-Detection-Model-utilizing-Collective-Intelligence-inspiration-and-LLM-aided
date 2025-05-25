import json
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
import csv
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast

accuracies = []
precisions = []
f1s = []
recalls = []

def parse_vector(vector):
    if isinstance(vector, str):
        try:
            vector = ast.literal_eval(vector) 
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing vector: {vector}, {e}")
            raise e
    
    while isinstance(vector, list) and isinstance(vector[0], list):
        vector = vector[0]
    
    if isinstance(vector, list):
        vector = [0.0 if isinstance(v, str) else v for v in vector]
    
    try:
        return np.array(vector, dtype=float)
    except ValueError as e:
        print(f"Error converting to numpy array: {vector}, {e}")
        raise e

def get_answer(df, id):
    beacon_doubt = np.zeros((1, 384))
    beacon_neutral = np.zeros((1, 384))
    beacon_support = np.zeros((1, 384))
    for index, item in df.iterrows():
        if item['id'] == id:
            item['dis_label'] = item['label']
            beacon_text = item['text_embedding']
            beacon_attitude = item['attitude']
            if item['sentiment'] == "质疑":
                beacon_doubt = item['embedding']
            if item['sentiment'] == "中立":
                beacon_neutral = item['embedding']
            if item['sentiment'] == "支持":
                beacon_support = item['embedding']            

    for index, row in df.iterrows():
        if row['id'] != id:
            text_distance = euclidean(beacon_text, row['text_embedding'])
            attitude_distance = euclidean(beacon_attitude, row['attitude'])
            
            if row['sentiment'] == "质疑":
                sentiment_distance = cosine_similarity(beacon_doubt, row['embedding'])
            if row['sentiment'] == "中立":
                sentiment_distance = cosine_similarity(beacon_neutral, row['embedding'])
            if row['sentiment'] == "支持":
                sentiment_distance = cosine_similarity(beacon_support, row['embedding'])
            df.at[index, 'text_distance'] = text_distance
            df.at[index, 'attitude_distance'] = attitude_distance
            df.at[index, 'sentiment_distance'] = json.dumps(sentiment_distance.tolist())
            

    nested_dict = {
        row["id"]: {
            "label": row["label"],
            "text" : None,
            "attitude" : None,
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


    parsed_feature_vectors = [parse_vector(v) for v in feature_vectors]

    feature_vectors = np.array(parsed_feature_vectors)


    n_clusters = 2  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(feature_vectors)

    for i, key in enumerate(keys):
        nested_dict[key]["cluster"] = int(labels[i])  
        


    true_labels = []
    predicted_labels = []

    for key, values in nested_dict.items():
        true_labels.append(nested_dict[key]["label"])
        predicted_labels.append(nested_dict[key]["cluster"])

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')  
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print("-------------------------------")
    print("ID:", id)
    print("预测准确率(Accuracy):", accuracy)
    print("精确率(Precision):", precision)
    print("召回率(Recall):", recall)
    print("F1-得分(F1-Score):", f1)
    return true_labels, predicted_labels
    
input_file = "/home/user1809/RenSifei/pythonFile/新实验/数据集/checked/多维度/topic0.json"

with open(input_file, 'r', encoding="utf-8") as f:
    data = json.load(f)

rows = []
for entry in data:
    for sentiment, embedding in entry["sentiment_embeddings"].items():
        rows.append({
            "id": entry["ID"],
            "label": entry["label"],
            "attitude" : entry["attitude"], 
            "text_embedding": entry["text_embedding"],
            "sentiment": sentiment,
            "embedding": embedding
        })
        
df = pd.DataFrame(rows)    

input_ids = input("输入ID：").strip()
id_list = input_ids.split(",")

for beacon_id in id_list:
    beacon_id = beacon_id.strip()
    true_labels, predicted_labels = get_answer(df, beacon_id)
    print("真实标签:", true_labels)
    print("预测标签:", predicted_labels)

output_file = "/home/user1809/RenSifei/pythonFile/新实验/数据集/checked/多维度/metrics_attitude_topic0.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "accuracies", "precisions", "f1s", "recalls"])

    for i in range(len(accuracies)):
        writer.writerow([id_list[i], accuracies[i], precisions[i], f1s[i], recalls[i]])


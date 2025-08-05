import json
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]
    return sentence_embedding.squeeze().numpy()

json_file_path = "input.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    text = item.get("text", "")
    embedding = get_sentence_embedding(text)
    item["embedding"] = embedding.tolist()

output_file_path = "output.json"
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


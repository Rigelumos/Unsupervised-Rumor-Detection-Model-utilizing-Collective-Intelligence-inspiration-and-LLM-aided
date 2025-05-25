import re
import json

input_json = "input.json"
with open(input_json, encoding='utf-8') as file1:
    json_data = json.load(file1)

data = [item['Answers'].replace('\n', '') for item in json_data]

keywords = ["质疑", "中立", "支持"]

def find_first_keyword(sentence, keywords):
    first_occurrence = len(sentence)
    first_keyword = None
    for keyword in keywords:
        index = sentence.find(keyword)
        if index != -1 and index < first_occurrence:
            first_occurrence = index
            first_keyword = keyword
    return first_keyword

all_keywords = [find_first_keyword(text, keywords) for text in data]

for i, entry in enumerate(json_data):
    entry['type'] = all_keywords[i % len(all_keywords)]

with open(input_json, 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)

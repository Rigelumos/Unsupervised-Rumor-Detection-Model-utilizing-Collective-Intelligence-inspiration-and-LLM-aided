import json
from collections import defaultdict

with open("input.json", encoding="utf-8") as file1:
    json_data = json.load(file1)

def divide(item):
    info_start = item['Prompt'].find('内容：') + len('内容：')
    info_end = item['Prompt'].find('请根据常识') if '请根据常识' in item['Prompt'] else item['Prompt'].find('以下是关于')
    info_content = item['Prompt'][info_start:info_end].strip()

    comment_start = item['Prompt'].find('以下是关于这则信息的评论：\n') + len('以下是关于这则信息的评论：\n')
    comment_end = item['Prompt'].find('请分析以上评论的感情色彩') if '请分析以上评论的感情色彩' in item['Prompt'] else len(item['Prompt'])
    comment_content = item['Prompt'][comment_start:comment_end].strip()
    return info_content, comment_content

formatted_data = []
for item in json_data:
    info_content, comment_content = divide(item)
    formatted_item = {
        "ID": item['ID'],
        "Comment_ID": item['Comment_ID'],
        "text": info_content,
        "comments": [
            {
                "comment_text": comment_content,
                "sentiment": item['type']
            }
        ],
        "label": 1
    }
    formatted_data.append(formatted_item)

grouped_data = defaultdict(lambda: {"ID": "", "text": "", "comments": [], "label": 0})
for item in formatted_data:
    item_id = item["ID"]
    if not grouped_data[item_id]["ID"]:
        grouped_data[item_id]["ID"] = item_id
        grouped_data[item_id]["text"] = item["text"]
        grouped_data[item_id]["label"] = item["label"]

    for comment in item["comments"]:
        grouped_data[item_id]["comments"].append({
            "Comment_ID": item["Comment_ID"],
            "comment_text": comment["comment_text"],
            "sentiment": comment["sentiment"]
        })

final_data = list(grouped_data.values())
with open("output.json", 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

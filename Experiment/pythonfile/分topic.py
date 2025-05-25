import json

def extract_by_topic(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    combined_data = data1 + data2

    topic_dict = {}
    for entry in combined_data:
        topic_value = entry.get("topic", "unknown")  
        if topic_value not in topic_dict:
            topic_dict[topic_value] = []
        topic_dict[topic_value].append(entry)

    for topic, entries in topic_dict.items():
        file_name = f"**"
        with open(file_name, 'w', encoding='utf-8') as topic_file:
            json.dump(entries, topic_file, ensure_ascii=False, indent=4)
        print(f"数据已保存到 {file_name}")

file1_path = '***'
file2_path = '***'
extract_by_topic(file1_path, file2_path)

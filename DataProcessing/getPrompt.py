import json
import os
import csv

input_folder = "input_folder"
output_file = "output.csv"

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['ID', 'comment_num', 'repost_num', 'like_num', 'Prompt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompt = "这是一则信息：\n"
        prompt += f"内容：{data['text']}\n\n"
        prompt += "请根据常识和上下文逻辑, 判断该信息的可信度, 即信息为真还是假还是不确定？如果你能判断它是真的或者假的, 准确率是多少？\n\n"
        prompt += "以下是关于这则信息的评论：\n"

        comments = [comment for comment in data['comments'] if comment['text'].strip() and comment['text'] != "："]
        for comment in comments:
            temp = prompt + comment['text']
            temp += "\n请逐条分析以上评论的感情色彩, 判断出每个评论发布者是否相信这则信息？并给出每个评论发布者对帖子的态度, 即是支持、质疑还是中立？请分点“1.”、“2.”...作答"
            writer.writerow({
                'ID': data['id'],
                'comment_num': data['comment_num'],
                'repost_num': data['repost_num'],
                'like_num': data['like_num'],
                'Prompt': temp
            })

        print(f"Processed {filename}")

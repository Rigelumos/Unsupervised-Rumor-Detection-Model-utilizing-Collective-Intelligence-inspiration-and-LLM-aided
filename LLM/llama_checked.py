from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import torch

model_path = "***"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda:0")
model.to(device)  

def remove_content_before(response, marker):

    marker_index = response.find(marker)
    prefix = "分析如下: \n"
    if marker_index != -1:
        response = response[marker_index + len(marker):].strip()
        return prefix + response
    else:
        return response


def process_questions(input_file, output_file,start_line, batch_size, marker):
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    questions_items = questions[start_line:]


    answers = []
    flag=start_line-1


    for item in questions_items:
        question = item['Prompt']
        response = Llama_response(question)
        cleaned_response = remove_content_before(response, marker)

        item['Answers'] = cleaned_response
        batch_answers.append(item)
        flag += 1
        print(item["Comment_ID"])

        if len(batch_answers) >= batch_size:
            write_to_file(output_file, batch_answers)
            answers.extend(batch_answers)
            batch_answers = []
            print("获得批次"+str(flag))
    if batch_answers:
        write_to_file(output_file, batch_answers)
        answers.extend(batch_answers)
   
    print(f"答案已保存至 {output_file}")

def write_to_file(output_file, batch_answers):
    try:
        with open(output_file, 'r+', encoding='utf-8') as f:
            try:
                existing_answers = json.load(f)
            except json.JSONDecodeError:
                existing_answers = []
            existing_answers.extend(batch_answers)
            f.seek(0)
            json.dump(existing_answers, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_answers, f, ensure_ascii=False, indent=4)

def Llama_response(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=5000)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

import pandas as pd
import json

start_line = 8440
batch_size = 5

marker = "请分析以上评论的感情色彩, 判断出评论发布者是否相信这则信息？并给出评论发布者对帖子的态度, 即是支持、质疑还是中立？请分点“1.”、“2.”...作答。"
process_questions(input_file, output_file, start_line , batch_size, marker)

import json
from collections import defaultdict

input_json1 = "fake.json"
with open(input_json1, encoding='utf-8') as file1:
    json_data1 = json.load(file1)

if 'attitude' in json_data1:
    del json_data1['attitude']

comment_attitude1 = []
for item in json_data1:
    comment_singel1 = []
    for comment in item['comments']:
        temp = comment['sentiment']
        comment_singel1.append(temp)
    comment_attitude1.append(comment_singel1)

input_json2 = "real.json"
with open(input_json2, encoding='utf-8') as file2:
    json_data2 = json.load(file2)

if 'attitude' in json_data2:
    del json_data2['attitude']

comment_attitude2 = []
for item in json_data2:
    comment_singel2 = []
    for comment in item['comments']:
        temp = comment['sentiment']
        comment_singel2.append(temp)
    comment_attitude2.append(comment_singel2)

def compute_attitude_distribution(comment_data):
    features = []
    labels = []
    scores = []

    for comments in comment_data:
        attitude_count = {'支持': 0, '质疑': 0, '中立': 0}
        total_comments = len(comments)

        for attitude in comments:
            if attitude in attitude_count:
                attitude_count[attitude] += 1

        attitude_percentages = {
            '支持': attitude_count['支持'] / total_comments,
            '质疑': attitude_count['质疑'] / total_comments,
            '中立': attitude_count['中立'] / total_comments
        }
        score = attitude_percentages['质疑'] * (a) + attitude_percentages['支持'] * b
        features.append([attitude_percentages['支持'], attitude_percentages['质疑'], attitude_percentages['中立']])
        scores.append([attitude_percentages['支持'] * c, attitude_percentages['质疑'] * (a), attitude_percentages['中立']*c])

    return scores

fake_scores = compute_attitude_distribution(comment_attitude1)
real_scores = compute_attitude_distribution(comment_attitude2)

for i, item in enumerate(json_data1):
    item['attitude'] = fake_scores[i]

for i, item in enumerate(json_data2):
    item['attitude'] = real_scores[i]

with open('fake_all.json', 'w', encoding='utf-8') as f:
    json.dump(json_data1, f, ensure_ascii=False, indent=4)

with open('real_all.json', 'w', encoding='utf-8') as f:
    json.dump(json_data2, f, ensure_ascii=False, indent=4)

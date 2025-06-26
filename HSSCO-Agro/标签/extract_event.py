import json
import os

interested_labels = {"CROP", "WE", "ENV", "INFRA", "ACTV", "DIS_PEST", "LOSS"}

input_file_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\实体\data2.1.json'
output_directory = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\实体'


texts_by_label = {label: [] for label in interested_labels}

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

    # Loop through each item in the JSON data
    for item in data:
        annotations = item.get('annotations', [])
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                labels = result.get('value', {}).get('labels', [])
                text = result.get('value', {}).get('text', '')
                for label in labels:
                    if label in interested_labels:
                        texts_by_label[label].append(text)

os.makedirs(output_directory, exist_ok=True)

for label, texts in texts_by_label.items():
    file_path = os.path.join(output_directory, f'{label}.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + '\n')

print(f"Extracted texts have been categorized and saved in '{output_directory}'")
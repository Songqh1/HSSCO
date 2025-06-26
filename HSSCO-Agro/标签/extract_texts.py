import json
import os

# 定义读取和写入文件的路径
json_file_path = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\周报.json'  # 替换为你的 JSON 文件路径
txt_dir_path = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_texts_by_label'  # 输出目录路径

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data_list = json.load(json_file)

# 初始化存储所有 text 的列表
texts = []

# 遍历 JSON 文件中的所有条目
if isinstance(data_list, list):
    for data in data_list:
        # 提取 "data" 中的 "text"
        text = data.get('data', {}).get('text', '')
        if text:
            texts.append(text)

# 将所有提取的文本合并为一个字符串
all_text = '\n'.join(texts)

# 创建目录（如果不存在）
os.makedirs(txt_dir_path, exist_ok=True)

# 定义文件路径并写入提取的文本
file_path = os.path.join(txt_dir_path, 'texts.txt')
with open(file_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(all_text)

print(f'Texts successfully extracted and saved to {file_path}')

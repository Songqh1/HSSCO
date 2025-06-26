import os

# 定义读取和写入文件的路径
input_txt_file_path = '/data/处理/texts.txt'  # 输入的 TXT 文件路径
output_txt_file_path = '/data/处理/sentences.txt'  # 输出的 TXT 文件路径

# 读取 TXT 文件内容
with open(input_txt_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 根据句号和分号分割文本
# 使用句号和分号作为分隔符，并保留分隔符的情况
import re
sentences = re.split(r'[。；]', text)

# 去除每个句子前后的空白，并过滤掉空字符串
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

# 将分割后的句子写入到新的 TXT 文件中
os.makedirs(os.path.dirname(output_txt_file_path), exist_ok=True)  # 创建目录（如果不存在）
with open(output_txt_file_path, 'w', encoding='utf-8') as file:
    for sentence in sentences:
        file.write(sentence + '\n')  # 每个句子占据一行

print(f'Sentences successfully extracted and saved to {output_txt_file_path}')

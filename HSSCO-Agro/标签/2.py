# import os
#
# def read_labels(folder_path):
#     labels = {}
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt") and not filename.startswith("."):
#             with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
#                 label = filename[:-4]
#                 labels[label] = f.read().splitlines()
#     return labels
#
# def process_files(input_file, labels, single_tag_file, multi_tag_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     with open(single_tag_file, 'w', encoding='utf-8') as f_single, \
#          open(multi_tag_file, 'w', encoding='utf-8') as f_multi:
#         for line in lines:
#             parts = line.strip().split(' ', 1)
#             if len(parts) == 2:
#                 tags, text_content = parts
#             else:
#                 tags = ''
#                 text_content = line.strip()
#
#             if len(tags.split('|')) == 1:
#                 f_single.write(f"{line}")
#             else:
#                 tags_set = set(tags.split('|')) if tags else set()
#                 for label, keywords in labels.items():
#                     for keyword in keywords:
#                         if keyword in text_content:
#                             tags_set.add(label)
#
#                 tags = '|'.join(sorted(tags_set))
#                 f_multi.write(f"{tags} {text_content}\n")
#
# def main():
#     folder_path = r'D:\Pycharm\bert-textcnn-for-multi-classfication\data\extracted_texts_by_label'
#     input_file = r'D:\Pycharm\bert-textcnn-for-multi-classfication\predictions.txt'
#     single_tag_file = 'single_tag_output.txt'
#     multi_tag_file = 'multi_tag_output.txt'
#
#     labels = read_labels(folder_path)
#     process_files(input_file, labels, single_tag_file, multi_tag_file)
#
# if __name__ == "__main__":
#     main()

def extract_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 去掉每行的空白字符
            line = line.strip()
            # 按照"|"分隔，取第二部分作为文本
            parts = line.split(' ', 1)
            if len(parts) > 1:
                # 将文本部分写入输出文件
                outfile.write(parts[1] + '\n')

# 调用函数，传入输入文件和输出文件的路径
input_file = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\extracted_texts_by_label\测试.txt'  # 输入文件路径
output_file = 'ceshi.txt'  # 输出文件路径
extract_text(input_file, output_file)

import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import numpy as np

# 指定BERT模型的配置文件和预训练权重文件的路径
config_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\chinese_bert_wwm_L-12_H-768_A-12\publish\\bert_config.json'  # 替换为你的BERT配置文件路径
checkpoint_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\chinese_bert_wwm_L-12_H-768_A-12\publish\\bert_model.ckpt'  # 替换为你的BERT模型权重文件路径
label_description_file = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\标签\标签描述.txt'  # 替换为你的标签描述信息文件路径
vocab_file = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\chinese_bert_wwm_L-12_H-768_A-12\\publish\\vocab.txt'
# 加载预训练的BERT模型
bert_model = load_trained_model_from_checkpoint(config_file=config_path,
                                               checkpoint_file=checkpoint_path)

# 读取词汇表文件
with open(vocab_file, 'r', encoding='utf-8') as file:
    vocab_dict = {line.strip(): idx + 1 for idx, line in enumerate(file)}

# 加载与之对应的分词器
tokenizer = Tokenizer(token_dict=vocab_dict)  # 替换为你的BERT词汇表文件路径

# 读取标签描述信息文件
with open(label_description_file, 'r', encoding='utf-8') as file:
    label_description = file.read()
    print(label_description)

# 对标签描述信息进行分词
tokenized_label_description = tokenizer.tokenize(label_description)
encoded_label_description = tokenizer.encode(tokenized_label_description)

# 创建一个与BERT模型输入匹配的mask和segmentation ID数组
input_mask = [1] * len(encoded_label_description)
segment_ids = [0] * len(encoded_label_description)  # 假设标签描述是第0段

# 将输入转换为模型所需的形状
encoded_label_description = np.array(encoded_label_description).reshape(1, -1)
input_mask = np.array(input_mask).reshape(1, -1)
segment_ids = np.array(segment_ids).reshape(1, -1)

# 获取BERT模型的输出
outputs = bert_model.predict([encoded_label_description, segment_ids, input_mask])

# 输出BERT编码的结果
print(outputs)
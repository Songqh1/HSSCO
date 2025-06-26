from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# 设置模型路径
model_path = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\huggingface\\bert-base-chinese'

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 添加一个线性层来减少嵌入向量的大小
down_layer = nn.Linear(768, 128)

def read_keywords_from_file(file_path):
    """
    从文件中读取关键词
    :param file_path: 关键词文件路径
    :return: 关键词列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines()]
    return keywords

def get_bert_embeddings(keywords):
    """
    获取BERT词嵌入
    :param keywords: 关键词列表
    :return: 词嵌入的字典 {词: 向量}
    """
    embeddings = {}

    for keyword in keywords:
        inputs = tokenizer(keyword, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取[CLS] token的嵌入，通常用于分类任务
        cls_embedding = outputs.last_hidden_state[0][0]
        # 应用线性层来减少嵌入向量的大小
        # cls_embedding = down_layer(cls_embedding.unsqueeze(0))  # 增加一个批次维度
        embeddings[keyword] = cls_embedding.squeeze(0).detach().numpy()  # 移除批次维度，转换为 NumPy 数组

    return embeddings

def save_embeddings_to_file(embeddings, file_path):
    """
    将词嵌入保存到文件
    :param embeddings: 词嵌入的字典 {词: 向量}
    :param file_path: 保存文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, vector in embeddings.items():
            vector_str = ' '.join(map(str, vector))
            f.write(f"{word} {vector_str}\n")

if __name__ == "__main__":
    # 读取关键词
    keywords_file_path = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\all_keywords.txt'  # 替换为实际的关键词文件路径
    keywords = read_keywords_from_file(keywords_file_path)

    # 获取BERT嵌入
    word_embeddings = get_bert_embeddings(keywords)
    for word, embedding in word_embeddings.items():
        print(f"Word: {word}, Embedding Shape: {embedding.shape}")

    # 保存到文件
    save_embeddings_to_file(word_embeddings, 'bert_word_embeddings.txt')
    print('成功保存')
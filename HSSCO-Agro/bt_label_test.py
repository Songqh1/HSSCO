import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import hamming_loss, classification_report
import pickle
from bt_label import BERTTextCNN
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import os

# 加载配置
class Config:
    bert_model_name = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\huggingface\\bert-base-chinese'
    max_len = 128
    batch_size = 16
    test_dataset_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\\test_sentence.txt'
    model_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\model\\bt_label_model.pth'

config = Config()

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_len]

        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.float), labels  # 返回 labels 用于 batch_labels

# 加载测试数据
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            labels.append(line[0].split('|'))
            texts.append(line[1] if len(line) > 1 else '')
    return texts, labels

# 加载多标签编码器
with open('D:\Pycharm\\bert-textcnn-for-multi-classfication\data\mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# 加载测试数据
test_texts, test_labels = load_data(config.test_dataset_path)
test_labels = mlb.transform(test_labels)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, config.max_len)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 词向量
def load_word_vectors(file_path):
    word_vecs = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            word_vecs[word] = vector
    return word_vecs
# 标签关键词
def load_label_keywords(directory_path):
    label_keywords = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            label = filename[:-4]  # Remove .txt extension
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                keywords = [line.strip() for line in file]
            label_keywords[label] = keywords
    return label_keywords
# 标签向量
def build_label_vectors(label_keywords, word_vecs):
    label_vecs = {}

    for label, keywords in label_keywords.items():
        vectors = [word_vecs.get(keyword) for keyword in keywords if keyword in word_vecs]

        if vectors:
            label_vecs[label] = np.mean(vectors, axis=0)
        else:
            vector_dim = next(iter(word_vecs.values())).shape[0] if word_vecs else 256
            label_vecs[label] = np.zeros(vector_dim)

    return label_vecs
word_vecs_file = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\标签\\bert_word_embeddings.txt'
keywords_dir = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\\new_label'

word_vecs = load_word_vectors(word_vecs_file)
label_keywords = load_label_keywords(keywords_dir)
label_vecs = build_label_vectors(label_keywords, word_vecs)
# 加载模型
label_vec_dim = len(next(iter(label_vecs.values())))  # 获取标签向量的维度
model = BERTTextCNN(num_classes=len(mlb.classes_), label_vec_dim=label_vec_dim)
model.load_state_dict(torch.load(config.model_path))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# 评估模型
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

        # 获取标签向量
        raw_labels = batch['labels'].cpu().numpy()
        batch_label_vecs = []
        for label_list in raw_labels:
            label_vectors = [label_vecs.get(str(label)) for label in label_list if str(label) in label_vecs]
            if label_vectors:
                avg_label_vec = torch.tensor(np.mean(label_vectors, axis=0), dtype=torch.float32)
            else:
                avg_label_vec = torch.zeros(label_vec_dim, dtype=torch.float32)
            batch_label_vecs.append(avg_label_vec)

        # 转换为张量并确保形状为 [batch_size, label_vec_dim]
        batch_label_vecs = torch.stack(batch_label_vecs).to('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = model(input_ids, attention_mask, batch_label_vecs)
        preds = outputs.cpu().numpy()
        labels = labels.cpu().numpy()

        # 将预测值二值化（使用阈值0.5）
        preds = (preds >= 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels)

# 计算自定义的完全匹配准确率
correct_count = 0
total_count = len(all_labels)

for i in range(total_count):
    # 比较预测和实际标签是否完全匹配
    if (all_preds[i] == all_labels[i]).all():
        correct_count += 1

# 输出分类报告
report = classification_report(all_labels, all_preds, target_names=mlb.classes_, digits=4)
print("Classification Report:\n", report)

# 计算汉明损失
hamming_loss_value = hamming_loss(all_labels, all_preds)
print(f"Hamming Loss: {hamming_loss_value:.4f}")

# 计算准确率
accuracy = correct_count / total_count
print(f"Accuracy (all labels must match): {accuracy:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import random
from tqdm import tqdm
import numpy as np
import os


# 配置类，包含训练所需参数
class Config:
    bert_model_name = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\huggingface\\bert-base-chinese'
    max_len = 128
    batch_size = 16
    learning_rate = 1e-5
    epochs = 1
    train_dataset_path = '/data/train_sentence.txt'
    val_dataset_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\\val_sentence.txt'
    test_dataset_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\data\\test_sentence.txt'

config = Config()

# 加载BERT的分词器
tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

# 数据集类，用于数据加载和处理
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

# 定义模型结构
class BERTTextCNN(nn.Module):
    def __init__(self, num_classes, label_vec_dim, dropout_rate=0.2, num_filters=256, kernel_sizes=[3, 4, 5]):
        super(BERTTextCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.units = self.bert.config.hidden_size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_fc = nn.Linear(num_filters * len(kernel_sizes), self.units)

        # 添加处理标签向量的层
        self.label_fc = nn.Linear(label_vec_dim, self.units)
        # 在模型初始化部分调整 fc1 的输入维度
        self.fc1 = nn.Linear(self.units + num_filters * len(kernel_sizes) + self.units, 256)  # 调整为组合后的维度
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, label_vecs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        embeddings = outputs.last_hidden_state[:, 1:-1]  # [batch_size, max_len-2, hidden_size]
        embeddings = embeddings.permute(0, 2, 1)  # [batch_size, hidden_size, max_len-2]

        cnn_outputs = [self.pool(torch.relu(conv(embeddings))).squeeze(-1) for conv in self.convs]
        cnn_features = torch.cat(cnn_outputs, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]

        # 处理标签向量
        label_features = torch.relu(self.label_fc(label_vecs))
        # 确保 label_features 的第一个维度与 cls_output 和 cnn_features 一致
        assert label_features.size(0) == cls_output.size(0)

        combined_features = torch.cat((cls_output, cnn_features, label_features), dim=1)
        combined_features = self.dropout(combined_features)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return torch.sigmoid(x)



# 数据加载函数
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            labels.append(line[0].split('|'))
            texts.append(line[1] if len(line) > 1 else '')
    return texts, labels

# 主程序
if __name__ == "__main__":
    # 加载数据
    train_texts, train_labels = load_data(config.train_dataset_path)
    val_texts, val_labels = load_data(config.val_dataset_path)

    # 打乱训练数据
    index = list(range(len(train_texts)))
    random.shuffle(index)
    train_texts = [train_texts[i] for i in index]
    train_labels = [train_labels[i] for i in index]

    # 标签二值化
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)
    train_labels = mlb.transform(train_labels)
    val_labels = mlb.transform(val_labels)

    # 保存编码器
    with open('D:\Pycharm\\bert-textcnn-for-multi-classfication\data\mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


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

    # 模型初始化
    label_vec_dim = len(next(iter(label_vecs.values())))  # 获取标签向量的维度
    model = BERTTextCNN(num_classes=len(mlb.classes_), label_vec_dim=label_vec_dim)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float('inf')  # 用于记录最小的验证损失
    model_save_path = 'D:\Pycharm\\bert-textcnn-for-multi-classfication\model\\bt_label_model.pth'


    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels, raw_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            # 使用 raw_labels 获取标签向量
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

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, batch_label_vecs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels, raw_labels in tqdm(val_loader, desc=f'Validation {epoch + 1}'):
                input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

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
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

        # 检查是否是最好的模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with validation loss: {best_loss:.4f}')

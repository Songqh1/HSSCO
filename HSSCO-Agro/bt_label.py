import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F
import pickle
import random
from tqdm import tqdm
import numpy as np
import os


# 设置随机种子以确保结果可重复
def set_seed(seed=42):
    random.seed(seed)  # Python 内置随机模块设置种子
    np.random.seed(seed)  # NumPy 设置种子
    torch.manual_seed(seed)  # PyTorch CPU 设置种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 设置种子
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 关闭自动优化


# 配置类，包含训练所需参数
class Config:
    bert_model_name = './huggingface/bert-base-chinese/'
    max_len = 128
    batch_size = 16
    learning_rate = 1e-5
    epochs = 20
    train_dataset_path = './data/train_sentence.txt'
    val_dataset_path = './data/val_sentence.txt'
    test_dataset_path = './data/test_sentence.txt'


config = Config()

# 加载BERT的分词器
tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)


# 数据集类，用于数据加载和处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, adj_matrices):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.adj_matrices = adj_matrices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        adj_matrix = self.adj_matrices[idx]
        # 转换为Tensor
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        adj_matrix = self._resize_adj_matrix(adj_matrix, self.max_len)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_len]

        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.float), labels, adj_matrix

    def _resize_adj_matrix(self, adj_matrix, target_size):
        current_size = adj_matrix.shape[0]

        # 如果当前大小大于目标大小，则裁剪
        if current_size > target_size:
            adj_matrix = adj_matrix[:target_size, :target_size]
        # 如果当前大小小于目标大小，则填充0
        elif current_size < target_size:
            pad_size = target_size - current_size
            pad = torch.zeros((pad_size, current_size), dtype=torch.float32)
            adj_matrix = torch.cat([adj_matrix, pad], dim=0)
            pad = torch.zeros((adj_matrix.shape[0], pad_size), dtype=torch.float32)
            adj_matrix = torch.cat([adj_matrix, pad], dim=1)

        return adj_matrix



class GCNLayer(nn.Module):
    def __init__(self, units=768, activation='relu'):
        super(GCNLayer, self).__init__()
        self.units = units
        self.activation = getattr(F, activation) if activation else None
        self.linear = nn.Linear(units, units)

    def forward(self, x, adj):
        batch_size = x.size(0)
        outputs = []

        for i in range(batch_size):
            features = x[i]  # (num_nodes, input_dim)
            adjacency = adj[i]  # (num_nodes, num_nodes)
            # 确保邻接矩阵和特征矩阵的数据类型相同
            adjacency = adjacency.float()  # 转换为浮点型
            features = features.float()  # 转换为浮点型

            num_nodes = adjacency.size(1)  # 获取邻接矩阵的节点数
            features = features.permute(1,0)  # 将形状从 (batch_size, feature_dim, num_nodes) 调整为 (batch_size, num_nodes, feature_dim)

            # GCN operation: H' = ReLU(AH * W + b)
            support = torch.matmul(adjacency, features)  # (num_nodes, input_dim)
            output = self.linear(support)  # (num_nodes, units)

            if self.activation:
                output = self.activation(output)  # Apply activation if specified

            outputs.append(output)
        return torch.stack(outputs)

class GCNWithAttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super(GCNWithAttentionFusion, self).__init__()
        self.attention = nn.Linear(768 + 768, 1)  # 注意力层

    def forward(self, a_gcn_features, a_label_features):
        # 拼接 GCN 特征和标签特征
        combined_features = torch.cat((a_gcn_features, a_label_features), dim=-1)

        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(combined_features), dim=-1)

        # 对 GCN 特征和标签特征应用注意力权重
        weighted_features = a_gcn_features * attention_weights + a_label_features * (1 - attention_weights)  # [16,768]

        return weighted_features

class GCNWithAttentionFusion1(nn.Module):
    def __init__(self, feature_dim):
        super(GCNWithAttentionFusion, self).__init__()
        self.attention = nn.Linear(feature_dim * 2, feature_dim)  # 注意力层

    def forward(self, a_gcn_features, a_label_features):
        # 拼接 GCN 特征和标签特征
        combined_features = torch.cat((a_gcn_features, a_label_features), dim=-1)  # [batch_size, 1536]

        # 计算注意力权重：输出的维度为 [batch_size, feature_dim]
        attention_weights = torch.softmax(self.attention(combined_features), dim=-1)  # [batch_size, 768]

        # 对 GCN 特征和标签特征应用注意力权重
        weighted_features = a_gcn_features * attention_weights + a_label_features * (1 - attention_weights)  # [batch_size, 768]

        return weighted_features

# 定义模型结构
class BERTTextCNN(nn.Module):
    def __init__(self, num_classes, label_vec_dim, gcn_units, dropout_rate=0.2, num_filters=256,
                 kernel_sizes=[3, 4, 5]):
        super(BERTTextCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.units = self.bert.config.hidden_size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])
        num_convs = 3
        # 定义卷积层时使用 kernel_size=1
        self.convs_label = nn.ModuleList(
            [nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1) for _ in range(num_convs)])

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_fc = nn.Linear(num_filters * len(kernel_sizes), self.units)
        self.att = GCNWithAttentionFusion(self.units)

        # 添加处理标签向量的层
        self.label_fc = nn.Linear(label_vec_dim, self.units)

        # 添加GCN层
        self.gcn_layer = GCNLayer(self.units, activation='relu')
        self.linear_projection = nn.Linear(2304, 768)

        # 定义最后的全连接层
        self.fc1 = nn.Linear(self.units, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, label_vecs, adj_matrix):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]

        embeddings = outputs.last_hidden_state[:, 1:-1]  # [batch_size, max_len-2, hidden_size]
        embeddings = embeddings.permute(0, 2, 1)  # [batch_size, hidden_size, max_len-2]

        cnn_outputs = [self.pool(torch.relu(conv(embeddings))).squeeze(-1) for conv in self.convs]
        cnn_features = torch.cat(cnn_outputs, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]

        label_vecs = label_vecs.unsqueeze(2)  # 将形状从 [batch_size,768] 转换为 [batch_size,768, sequence_length]
        # 使用卷积层
        label_outputs = [self.pool(torch.relu(conv(label_vecs))).squeeze(-1) for conv in self.convs_label]
        label_features = torch.cat(label_outputs, dim=1)  # [batch_size,768]

        # 处理GCN特征
        seq_len = embeddings.size(2)  # 获取 embeddings 的序列长度
        adj_matrix_size = adj_matrix.size(1)  # 获取邻接矩阵的节点数

        # 确保只有在需要时才进行填充
        if seq_len != adj_matrix_size:
            if seq_len > adj_matrix_size:
                embeddings = embeddings[:, :, :adj_matrix_size]  # 截断 embeddings
            else:
                pad_size = adj_matrix_size - seq_len
                if pad_size > 0:  # 仅在 pad_size 为正时进行填充
                    pad = torch.zeros((embeddings.size(0), embeddings.size(1), pad_size), dtype=torch.float32).to(
                        embeddings.device)
                    embeddings = torch.cat([embeddings, pad], dim=2)  # 填充 embeddings
        gcn_features = self.gcn_layer(embeddings, adj_matrix)
        gcn_features = torch.mean(gcn_features, dim=1)

        # 组合特征
        a_gcn_features = torch.cat((cls_output, cnn_features, gcn_features), dim=1)
        a_gcn_features = self.linear_projection(a_gcn_features)
        combined_features = self.att(a_gcn_features, label_features)

        combined_features = self.dropout(combined_features)
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(1)
        return x


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
    set_seed(42)
    # 载入邻接矩阵
    train_adj_matrices = np.load(
        './data/adjacency_matrices_train.npy',
        allow_pickle=True)
    val_adj_matrices = np.load(
        './data/adjacency_matrices_val.npy',
        allow_pickle=True)

    # 加载数据
    train_texts, train_labels = load_data(config.train_dataset_path)
    val_texts, val_labels = load_data(config.val_dataset_path)

    # 打乱训练数据
    index = list(range(len(train_texts)))
    random.shuffle(index)
    train_texts = [train_texts[i] for i in index]
    train_labels = [train_labels[i] for i in index]
    train_adj_matrices = [train_adj_matrices[i] for i in index]

    # 标签二值化
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)
    train_labels = mlb.transform(train_labels)
    val_labels = mlb.transform(val_labels)

    # 保存编码器
    with open('./data/mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)


    def custom_collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        raw_labels = [item[3] for item in batch]
        adj_matrices = torch.stack([torch.tensor(item[4], dtype=torch.float32) for item in batch])  # 转换为 torch.Tensor

        return input_ids, attention_mask, labels, raw_labels, adj_matrices


    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_len, train_adj_matrices)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_len, val_adj_matrices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)


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


    word_vecs_file = './标签/bert_word_embeddings.txt'
    keywords_dir = './data/new_labels'

    word_vecs = load_word_vectors(word_vecs_file)
    label_keywords = load_label_keywords(keywords_dir)
    label_vecs = build_label_vectors(label_keywords, word_vecs)

    # 模型初始化
    label_vec_dim = len(next(iter(label_vecs.values())))  # 获取标签向量的维度
    model = BERTTextCNN(num_classes=len(mlb.classes_), label_vec_dim=label_vec_dim, gcn_units=768)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 损失函数和优化器
    criterion = nn.BCELoss()
    # criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float('inf')  # 用于记录最小的验证损失
    model_save_path = '/home/zhaohua/songqinghua/bert-textcnn-for-multi-classfication/model/seed_bt_labelcnn_gcn_model.pth'

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels, raw_labels, adj_matrix in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # print("111adj_matrix:",adj_matrix.shape)
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            adj_matrix = adj_matrix.to('cuda' if torch.cuda.is_available() else 'cpu')

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
            outputs = model(input_ids, attention_mask, batch_label_vecs, adj_matrix)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels, raw_labels, adj_matrix in tqdm(val_loader,
                                                                                  desc=f'Validation {epoch + 1}'):
                input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                adj_matrix = adj_matrix.to('cuda' if torch.cuda.is_available() else 'cpu')

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

                outputs = model(input_ids, attention_mask, batch_label_vecs, adj_matrix)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

        # 检查是否是最好的模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with validation loss: {best_loss:.4f}')


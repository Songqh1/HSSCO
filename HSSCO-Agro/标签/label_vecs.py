import numpy as np
import os

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
                keywords = [line.strip() for line in file if line.strip()]
            label_keywords[label] = keywords
    return label_keywords

# 标签向量
def build_label_vectors(label_keywords, word_vecs):
    label_vecs = {}

    for label, keywords in label_keywords.items():
        if not keywords:
            vector_dim = next(iter(word_vecs.values()), np.zeros(256)).shape[0]
            label_vecs[label] = np.zeros(vector_dim)
        else:
            vectors = [word_vecs.get(keyword) for keyword in keywords if keyword in word_vecs]

            if vectors:
                label_vecs[label] = np.mean(vectors, axis=0)
            else:
                vector_dim = next(iter(word_vecs.values())).shape[0] if word_vecs else 256
                label_vecs[label] = np.zeros(vector_dim)

    return label_vecs

word_vecs_file = 'bert_word_embeddings.txt'
keywords_dir = '../data/new_label'

word_vecs = load_word_vectors(word_vecs_file)
label_keywords = load_label_keywords(keywords_dir)
label_vecs = build_label_vectors(label_keywords, word_vecs)

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAttentionLayer(nn.Module):
    def __init__(self, label_vecs, hidden_dim):
        super(LabelAttentionLayer, self).__init__()
        self.label_vecs = nn.ParameterList(
            [nn.Parameter(torch.tensor(vec, dtype=torch.float)) for vec in label_vecs.values()])
        self.hidden_dim = hidden_dim

    def forward(self, lstm_output):
        # lstm_output: tensor of shape (batch_size, sequence_length, hidden_dim)
        # for bidirectional LSTM, lstm_output will have shape (batch_size, sequence_length, 2 * hidden_dim)

        # Calculate attention scores for each label vector
        attention_scores = []
        for label_vec in self.label_vecs:
            # Expand label vector to match the batch size and sequence length
            label_vec = label_vec.unsqueeze(0).unsqueeze(0).expand(lstm_output.size(0), lstm_output.size(1),
                                                                   self.hidden_dim)

            # Compute the attention score using a simple dot product
            score = torch.bmm(lstm_output, label_vec.transpose(1, 2))
            attention_scores.append(score)

        # Normalize attention scores using softmax
        attention_scores = torch.stack(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of LSTM outputs using attention weights
        # attention_weights: (batch_size, sequence_length, num_labels)
        # lstm_output: (batch_size, sequence_length, hidden_dim)
        weighted_output = torch.bmm(attention_weights, lstm_output)

        # Combine forward and backward attention if lstm_output is from a bidirectional LSTM
        if lstm_output.size(-1) == 2 * self.hidden_dim:
            lstm_output = lstm_output.view(lstm_output.size(0), -1, 2, self.hidden_dim)
            weighted_output = weighted_output.view(weighted_output.size(0), -1, 2, self.hidden_dim)

            # Combine forward and backward weighted outputs
            weighted_output = torch.cat((weighted_output[:, :, 0, :], weighted_output[:, :, 1, :]), dim=-1)

        return weighted_output


# Example usage:
hidden_dim = 256  # Dimension of the LSTM hidden state
label_vecs = label_vecs  # Dictionary loaded from the previous code snippet

# Initialize the label attention layer
label_attention_layer = LabelAttentionLayer(label_vecs, hidden_dim)

# Dummy LSTM output (batch_size, sequence_length, 2 * hidden_dim)
lstm_output = torch.randn(32, 50, 2 * hidden_dim)

# Get the text representation after label attention
text_representation = label_attention_layer(lstm_output)
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# 定义文件夹和文件路径
input_folder = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_texts_by_label'  # 输入文件夹路径
output_folder = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_keywords'  # 输出文件夹路径


# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有文件路径
files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.txt')]

for file_path in files:
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = file.read().splitlines()

    # 文本向量化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(keywords)

    # 计算k距离图
    k = 5  # 通常选择较小的k值，如数据维度的两倍
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = distances[:, -1]

    # 绘制k距离图
    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(k_distances), marker='o')
    plt.title(f'k-Distance Graph for {os.path.basename(file_path)}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'k_distance_graph_{os.path.basename(file_path)}.png'))
    plt.close()

    # 使用合适的eps进行DBSCAN聚类
    eps = 0.2  # 这个值需要通过k距离图观察确定
    min_samples = 4

    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
    labels = db.labels_

    # 找出每个簇的中心关键词
    unique_labels = set(labels)
    cluster_keywords = {}

    for label in unique_labels:
        if label == -1:
            # 跳过噪声
            continue

        # 获取该簇的关键词
        cluster_indices = np.where(labels == label)[0]
        cluster_tfidf_matrix = X[cluster_indices]

        # 计算簇内各关键词与簇中心的相似度
        mean_vector = np.mean(cluster_tfidf_matrix, axis=0)
        similarities = cosine_similarity(cluster_tfidf_matrix, mean_vector.reshape(1, -1)).flatten()

        # 获取前10个关键词
        sorted_indices = np.argsort(-similarities)
        top_keywords = [keywords[i] for i in sorted_indices[:10]]

        cluster_keywords[label] = top_keywords

    # 保存结果到txt文件
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for label, top_keywords in cluster_keywords.items():
            file.write(f'Cluster {label}:\n')
            for keyword in top_keywords:
                file.write(f'  {keyword}\n')
            file.write('\n')

print('聚类完成，并已保存到文件中。')

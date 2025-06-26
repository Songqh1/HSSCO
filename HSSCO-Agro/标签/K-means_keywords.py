import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# 自定义分词函数
def jieba_tokenizer(text):
    return list(jieba.cut(text))


# 读取文件夹中的所有txt文件
def read_files_from_folder(folder_path):
    texts = []
    file_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read().strip().split('\n'))
                file_names.append(file_name)

    return texts, file_names



def extract_and_save_keywords(texts, file_names, output_folder, n_clusters=10, top_n=5):
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)

    for i, (events, file_name) in enumerate(zip(texts, file_names)):
        if len(events) < 2:
            # 如果事件数量少于2，跳过聚类
            continue

        # 将事件文本向量化
        tfidf_matrix = vectorizer.fit_transform(events)
        tfidf_matrix = normalize(tfidf_matrix)  # 归一化

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(tfidf_matrix)

        # 获取每个聚类中心
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # 提取每个聚类的前10个不重复事件作为关键词
        top_keywords = set()
        for cluster_num in range(n_clusters):
            cluster_indices = np.where(labels == cluster_num)[0]
            cluster_matrix = tfidf_matrix[cluster_indices]
            cluster_center = cluster_centers[cluster_num]

            # 计算每个事件与聚类中心的相似度
            similarity_scores = cluster_matrix.dot(cluster_center)
            similarity_scores = similarity_scores.flatten()  # 转换为一维数组

            # 获取最相关的10个事件
            top_event_indices = np.argsort(similarity_scores)[-top_n:]  # 获取最相关的10个事件

            for idx in top_event_indices:
                top_keywords.add(events[cluster_indices[idx]])
                if len(top_keywords) >= n_clusters * top_n:
                    break
            if len(top_keywords) >= n_clusters * top_n:
                break

        # 如果提取到的关键词少，则补充
        top_keywords = list(top_keywords)
        if len(top_keywords) < n_clusters * top_n:
            additional_keywords = set(events) - set(top_keywords)
            top_keywords.extend(list(additional_keywords)[:n_clusters * top_n - len(top_keywords)])

        # 保存到文件
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for keyword in top_keywords[:n_clusters * top_n]:
                file.write(keyword + '\n')


# 文件夹路径配置
input_folder = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_texts_by_label'  # 输入文件夹路径
output_folder = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_keywords'  # 输出文件夹路径

# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

# 读取文件夹中的文件
texts, file_names = read_files_from_folder(input_folder)

# 提取关键词并保存
extract_and_save_keywords(texts, file_names, output_folder)

print("关键词提取和保存完成。")

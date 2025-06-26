import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# 读取目录下的所有文件
def read_files(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            files.append(os.path.join(directory, filename))
    return files

# 写入聚类结果到文件
def write_clusters(cluster_keywords, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for keyword in cluster_keywords:
            file.write(f"{keyword}\n")

# 主函数
def main():
    input_directory = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\extracted_texts_by_label'  # 输入文件目录
    output_directory = 'D:\\Pycharm\\bert-textcnn-for-multi-classfication\\data\\new_label'  # 输出文件目录

    # 创建输出目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 读取目录下的所有标签文件
    files = read_files(input_directory)

    for file_path in files:
        # 读取每个文件中的关键词
        with open(file_path, 'r', encoding='utf-8') as file:
            keywords = [line.strip() for line in file if line.strip()]
            label = os.path.basename(file_path).split('.')[0]  # 假设文件名就是标签

            # 使用TF-IDF向量化关键词
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(keywords)

            # 将稀疏矩阵转换为密集矩阵
            X = X.toarray()

            # 应用 MiniBatchKMeans 聚类
            minibatch_kmeans = MiniBatchKMeans(n_clusters=50, random_state=0)
            minibatch_kmeans.fit(X)
            labels = minibatch_kmeans.labels_

            # 获取每个群组的关键词
            unique_labels = np.unique(labels)
            cluster_keywords = [keywords[labels.tolist().index(label)] for label in unique_labels if labels.tolist().index(label) != -1]

            # 写入聚类结果到文件
            output_file = os.path.join(output_directory, f'{label}.txt')
            write_clusters(cluster_keywords, output_file)
            print(f"Clustered keywords for label '{label}' have been written to {output_file}")

if __name__ == "__main__":
    main()
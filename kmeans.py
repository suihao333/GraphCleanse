import torch
from sklearn.cluster import KMeans
import numpy as np
import pickle

# 用于读取的.pkl文件的路径
file_path = '/data/AIDS/AIDS/bkd/graph_emb.pkl'
with open(file_path, 'rb') as file:
    GCL_graph_emb_2000 = pickle.load(file)

num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(np.array(GCL_graph_emb_2000))

with open('/data/AIDS/AIDS/bkd/GCL_graph_kmeans_labels.pkl', 'wb') as file:
    pickle.dump(kmeans.labels_, file)
import numpy as np
from numpy.core.function_base import _logspace_dispatcher
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import scipy.sparse as sp

import torch
from torch.nn.functional import threshold

# 转换成独热编码


def encode_onehot(labels):
    onehot_encoder = OneHotEncoder()
    labels_onehot = onehot_encoder.fit_transform(labels).toarray()
    return labels_onehot


def encode_onehot_labels(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def alert_edge(labels):
    # weight = (0.025,0.025,0.025,0.2,0.2,0.025,0.1,0.2,0.2)
    # THRESHOLD = 0.65
    Alert_df = pd.DataFrame(labels)
    # alert_threshold = 0.0
    for i in range(Alert_df.shape[0]):
        for k in range(Alert_df.shape[0]-i-1):
            print(i, i+k+1)
    #         if alert_threshold >= THRESHOLD:
            file = r'./data/alert/alert.similarity'
            with open(file, 'a+') as f:
                f.write(Alert_df.iloc[i, 8]+','+Alert_df.iloc[i+k+1, 8]+"\n")


def load_data(path="./data/alert/", dataset="alert"):
    """Load alert dataset"""
    print('Loading {} dataset...'.format(dataset))
    # 读取警报文件
    idx_features_labels = np.loadtxt("{}{}.content".format(
        path, dataset), dtype=np.dtype(str), delimiter=',', skiprows=1)
    # 将警报类型转换成独热编码
    labels = encode_onehot_labels(idx_features_labels[:, -1])  # 警报类型
    # 将每篇文献的编号提取出来
    idx = np.array(idx_features_labels[:, 8], dtype=np.int32)
    # 警报图的边
    # alert_edge(idx_features_labels[1979:4826,:-1])
    # 将警报文件转换成独热编码
    idx_features_labels = encode_onehot(idx_features_labels[:, :-1])
    features = sp.csr_matrix(idx_features_labels[:, :-1], dtype=np.float32)
    """生成无向图"""
    # 对文献的编号构建字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 读取edges.txt文件
    edges_unordered = np.loadtxt("{}{}.similarity".format(
        path, dataset), dtype=np.int32, delimiter=',')
    # 生成图的边，（x,y）其中x、y都是为以警报id为索引得到的值
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 生成领接矩阵，生成的矩阵为稀疏矩阵，对应的行和列坐标分别为边的两个点，该步骤之后得到的是一个有向图
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        labels.shape[0], labels.shape[0]), dtype=np.float32)
    # 无向图的领接矩阵是对称的，因此需要将上面得到的矩阵转换为对称的矩阵，从而得到无向图的领接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    idx_train = range(3300)
    idx_val = range(3300, 5000)
    idx_test = range(5000, 6900)

    # 将特征转换为tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    # print(labels.max().item() + 1)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('Finish loading')

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """归一化函数"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """系数矩阵转稀疏丈量函数"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

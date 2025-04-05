import os
import heapq
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random

import torch_geometric
from sklearn.metrics import f1_score
import pickle
import copy
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, CoraFull, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from sklearn import preprocessing

"""base_num_dic = {'Amazon_clothing': 20, 'Amazon_electronics': 17, 'cora-full': 30, 'coauthorCS': 0, 'Computers': 0}  # num of classes
pretrain_split_dict = {'train': 400, 'dev': 50, 'test': 50}
metatrain_split_dict = {  # num of classes
    'Amazon_clothing': {'train': 30, 'test': 27},
    'Amazon_electronics': {'train': 100, 'test': 50},
    'cora-full': {'train': 20, 'test': 20},
    'coauthorCS': {'train': 5, 'test': 10},
    'Computers': {'train': 5, 'test': 5}
}"""

base_num_dic = {'Amazon_clothing': 47, 'Amazon_electronics': 17, 'cora-full': 40  , 'coauthorCS': 0, 'Computers': 0}  # num of classes
pretrain_split_dict = {'train': 400, 'dev': 50, 'test': 50}
metatrain_split_dict = {  # num of classes
    'Amazon_clothing': {'train': 20, 'test': 10},
    'Amazon_electronics': {'train': 100, 'test': 50},
    'cora-full': {'train':20, 'test': 10},
    'coauthorCS': {'train': 5, 'test': 10},
    'Computers': {'train': 5, 'test': 5},
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_curr_data(data, curr_id):
    """
    从原始图数据中提取指定节点的子图并封装为新的 Data 对象。

    参数:
    - data: 原始图数据对象，包含 .x 和 .edge_index。
    - curr_train_id: 需要提取的节点 ID 列表。

    返回:
    - curr_data: 包含提取的子图特征和边的新的 Data 对象。
    """
    # 提取curr_id对应的子图并重新索引节点
    edge_index_curr, mapping = subgraph(curr_id, data.edge_index, relabel_nodes=True)

    # 提取curr_id节点的特征
    x_curr = data.x[curr_id]

    # 创建新的Data对象并返回
    curr_data = Data(x=x_curr, edge_index=edge_index_curr)

    return curr_data


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':
        # 注意：softmax 需要指定维度；这里假设对最后一个维度进行softmax
        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


def load_raw_data(dataset_source):
    base_num = base_num_dic[dataset_source]
    if dataset_source in ['Amazon_clothing', 'Amazon_electronics']:
        n1s = []
        n2s = []
        for line in open("./few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_all_nodes = max(max(n1s), max(n2s)) + 1
        edges = torch.LongTensor([n1s, n2s])

        data_train = sio.loadmat("./few_shot_data/{}_train.mat".format(dataset_source))
        # train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

        data_test = sio.loadmat("./few_shot_data/{}_test.mat".format(dataset_source))
        # class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        labels = np.zeros((num_all_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_all_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        # Create PyG Data object
        features = torch.FloatTensor(features)
        pyg_labels = torch.LongTensor(labels)
        data = Data(x=features, edge_index=edges, y=pyg_labels)

        node_list = []
        for _, v in id_by_class.items():
            node_list.append(len(v))

        large_res_idex = heapq.nlargest(base_num, enumerate(node_list), key=lambda x: x[1])

        all_id = [i for i in range(len(node_list))]
        base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
        novel_id = list(set(all_id).difference(set(base_id)))

    elif dataset_source == 'cora-full':

        data = CoraFull(root='./data/Planetoid', transform=NormalizeFeatures())
        labels = data.data.y
        num_all_nodes = data.data.x.size(0)
        print(f"num_all_nodes:{num_all_nodes}" )

        # Create id_by_class dictionary
        class_list = labels.unique().tolist()

        id_by_class = {i: [] for i in class_list}
        for idx, label in enumerate(labels):
            id_by_class[label.item()].append(idx)

        # Calculate the number of nodes per class
        node_list = [len(v) for v in id_by_class.values()]

        # Get base and novel class IDs (e.g., selecting top classes based on number of nodes)
        base_num = base_num_dic[dataset_source]
        large_res_idex = heapq.nlargest(base_num, enumerate(node_list), key=lambda x: x[1])

        all_id = [i for i in range(len(node_list))]
        base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
        novel_id = list(set(all_id).difference(set(base_id)))

    elif dataset_source == 'coauthorCS':

        data = Coauthor(root='./data/CS', name='CS')
        labels = data.data.y
        num_all_nodes = data.data.x.size(0)

        # Create id_by_class dictionary
        class_list = labels.unique().tolist()

        id_by_class = {i: [] for i in class_list}
        for idx, label in enumerate(labels):
            id_by_class[label.item()].append(idx)

        # Calculate the number of nodes per class
        node_list = [len(v) for v in id_by_class.values()]

        # Get base and novel class IDs (e.g., selecting top classes based on number of nodes)
        base_num = base_num_dic[dataset_source]
        large_res_idex = heapq.nlargest(base_num, enumerate(node_list), key=lambda x: x[1])

        all_id = [i for i in range(len(node_list))]
        base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
        novel_id = list(set(all_id).difference(set(base_id)))

    elif dataset_source == 'Computers':

        data = Amazon(root='./data', name='Computers')
        labels = data.data.y
        num_all_nodes = data.data.x.size(0)

        # Create id_by_class dictionary
        class_list = labels.unique().tolist()

        id_by_class = {i: [] for i in class_list}
        for idx, label in enumerate(labels):
            id_by_class[label.item()].append(idx)

        # Calculate the number of nodes per class
        node_list = [len(v) for v in id_by_class.values()]

        # Get base and novel class IDs (e.g., selecting top classes based on number of nodes)
        base_num = base_num_dic[dataset_source]
        large_res_idex = heapq.nlargest(base_num, enumerate(node_list), key=lambda x: x[1])

        all_id = [i for i in range(len(node_list))]
        base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
        novel_id = list(set(all_id).difference(set(base_id)))



    return data, id_by_class, base_id, novel_id, node_list, num_all_nodes


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# Convert scipy sparse matrix to torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_base_data(base_id, id_by_class, labels):
    train_num = pretrain_split_dict['train']
    dev_num = pretrain_split_dict['dev']
    test_num = pretrain_split_dict['test']
    pretrain_idx = []
    predev_idx = []
    pretest_idx = []
    random.shuffle(base_id)

    for cla in base_id:
        node_idx = id_by_class[cla]
        random.shuffle(node_idx)
        pretrain_idx.extend(node_idx[: train_num])
        predev_idx.extend(node_idx[train_num: train_num + dev_num])
        pretest_idx.extend(node_idx[train_num + dev_num: train_num + dev_num + test_num])

    base_train_label = labels[pretrain_idx]
    base_dev_label = labels[predev_idx]
    base_test_label = labels[pretest_idx]

    base_train_id = sorted(set(base_train_label))
    base_dev_id = sorted(set(base_dev_label))
    base_test_id = sorted(set(base_test_label))

    return pretrain_idx, predev_idx, pretest_idx, base_train_label, base_dev_label, \
        base_test_label, base_train_id, base_dev_id, base_test_id


def simgrace_data(base_id, id_by_class, labels):
    pretrain_idx = []
    random.shuffle(base_id)
    for cla in base_id:
        node_idx = id_by_class[cla]
        random.shuffle(node_idx)
        pretrain_idx.extend(node_idx)

    train_label = labels[pretrain_idx]
    train_id = sorted(set(train_label))

    return pretrain_idx, train_label, train_id


def split_novel_data(novel_id, dataset_source):
    split_dict = metatrain_split_dict[dataset_source]
    random.shuffle(novel_id)
    metatrain_class_num = split_dict['train']
    novel_train_id = novel_id[: metatrain_class_num]
    novel_test_id = novel_id[metatrain_class_num:]

    return novel_train_id, novel_test_id

def check_id(id_by_class, base_id, novel_train_id, novel_test_id, n_way):
    # 检查novel_test_id中是否有类别的节点个数小于15，若存在则抛出错误
    novel_test_classes_to_remove = [cla for cla in novel_test_id if len(id_by_class[cla]) < 35]
    for cla in novel_test_classes_to_remove:
        novel_test_id.remove(cla)
        print(f"cla {cla} is removed!")

    # 检查novel_train_id中是否有类别的节点小于25，\
    # hint：number 30 is chosen as val_task require at least m_query samples and train task require at least k_shot+m_query samples.\
    # Thus, 30 = 2*(m_query + k_shot) for train and test
    # 如存在则进行下面操作: delete this class, 更新novel_train_id

    novel_classes_to_remove = [cla for cla in novel_train_id if len(id_by_class[cla]) < 30]
    base_classes_to_remove = [cla for cla in base_id if len(id_by_class[cla]) < 30]
    # 从novel_train_id中删除记录的类别
    for cla in novel_classes_to_remove:
        novel_train_id.remove(cla)
        print(f"cla {cla} is removed!")

    for cla in base_classes_to_remove:
        base_id.remove(cla)
        print(f"cla {cla} is removed!")
    
    remainder = len(novel_train_id) % n_way
    if remainder != 0:
        classes_to_move = random.sample(novel_train_id, remainder)
        for cla in classes_to_move:
            novel_train_id.remove(cla)
            base_id.append(cla)
            print(f"cla {cla} is moved from novel_train_id to base_id to ensure divisibility by {n_way}")

    return base_id, novel_train_id, novel_test_id


def split_novel_train_val(id_by_class, novel_train_id, val_size=30):
    novel_train_pool = {}
    novel_val_pool = {}

    for cla in novel_train_id:
        node_idx = id_by_class[cla]
        if len(node_idx) < val_size:
            print(f"Class {cla} doesn't have enough samples for validation. Only has {len(node_idx)} samples.")
            # 如果样本数量不够，所有样本划入训练集
            novel_train_pool[cla] = node_idx[:]
        else:
            random.shuffle(node_idx)
            novel_val_pool[cla] = node_idx[:val_size]  # First 30 for validation
            novel_train_pool[cla] = node_idx[val_size:]  # Remaining for training

    return novel_train_pool, novel_val_pool


def split_train_val(id_by_class, class_id, val_size=30):
    train_pool = {}
    val_pool = {}

    for cla in class_id:
        node_idx = id_by_class[cla]
        if len(node_idx) < val_size:
            print(f"Class {cla} doesn't have enough samples for validation size. Only has {len(node_idx)} samples.")
            # 如果样本数量不够，所有样本划入训练集
            if len(node_idx) < 15:
                raise ValueError(f"Sample fails! Query number must be less than 10 for class{cla}")
            else:
                print(f"Sample Succeeds! Class {cla}  has a validation pool with only 15 samples.")
                val_pool[cla] = node_idx[:15]
                train_pool[cla] = node_idx[15:]
        else:
            random.shuffle(node_idx)
            val_pool[cla] = node_idx[:val_size]  # First 30 for validation
            train_pool[cla] = node_idx[val_size:]  # Remaining for training

    return train_pool, val_pool


def split_base_train_val(id_by_class, base_id, val_size=30):
    base_train_pool = {}
    base_val_pool = {}

    for cla in base_id:
        node_idx = id_by_class[cla]
        if len(node_idx) < val_size:
            print(f"Class {cla} doesn't have enough samples for validation. Only has {len(node_idx)} samples.")
            # 如果样本数量不够，所有样本划入训练集
            base_train_pool[cla] = node_idx[:]
        else:
            random.shuffle(node_idx)
            base_val_pool[cla] = node_idx[:val_size]  # First 30 for validation
            base_train_pool[cla] = node_idx[val_size:]  # Remaining for training
    return base_train_pool, base_val_pool


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_base_adj(adj, base_id, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)

    mask = []
    for i in range(I.shape[1]):
        if labels[I[0, i]] in base_id and labels[I[1, i]] in base_id:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)

    I_base = I[:, mask]
    V_base = V[mask]

    base_adj = torch.sparse_coo_tensor(I_base, V_base, (dim_base, dim_base)).coalesce()

    return base_adj


def get_incremental_adj(adj, base_id, novel_id_support, novel_id_query, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    novel_idx = np.append(novel_id_support, novel_id_query)

    I = I.cuda()
    novel_idx = torch.from_numpy(novel_idx).cuda()
    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
                (I[0, i] in novel_idx and I[1, i] in novel_idx):
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def task_generator(class_id, class_pool, n_way, k_shot, m_query):
    # sample class indices
    class_selected = class_id
    id_support = []
    id_query = []
    labels_support = []
    labels_query = []

    for cla in class_selected:
        num_samples = len(class_pool[cla])
        if num_samples < (k_shot + m_query):
            print(f"Error: Not enough samples for label {cla}. Required: {k_shot + m_query}, Available: {num_samples}")

        temp = random.sample(class_pool[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])
        labels_support.extend([cla] * k_shot)
        labels_query.extend([cla] * m_query)

    labels_support = torch.tensor(labels_support)
    labels_query = torch.tensor(labels_query)

    return np.array(id_support), np.array(id_query), labels_support, labels_query, class_selected


def incremental_train_task_generator(id_by_class, class_pool, n_way, k_shot, m_query, base_id, novel_id, base_id_original):
    """print('base_id:',base_id)
    print('novel_id:',novel_id)"""
    # sample class indices
    base_id_delete = random.sample(base_id_original, len(base_id_original) // 2)
    base_id_delete = set(base_id_delete)
    base_class_selected = [b for b in base_id if b not in base_id_delete] # This step aims to balance the expectation of getting support data
    novel_class_selected = random.sample(novel_id, n_way)

    novel_id_support = []
    novel_id_query = []
    base_id_support = []
    base_id_query = []

    novel_labels_support = []
    novel_labels_query = []
    base_labels_support = []
    base_labels_query = []

    # Sample for novel classes
    for cla in novel_class_selected:
        temp = random.sample(class_pool[cla], k_shot + m_query)
        novel_id_support.extend(temp[:k_shot])
        novel_id_query.extend(temp[k_shot:])
        novel_labels_support.extend([cla] * k_shot)
        novel_labels_query.extend([cla] * m_query)

    # Since classes in base_id come from class_pool at the end of last incremental session
    # Sample for base classes
    for cla in base_class_selected:
        temp = random.sample(class_pool[cla], k_shot + m_query)
        base_id_support.extend(temp[:k_shot])
        base_id_query.extend(temp[k_shot:])
        base_labels_support.extend([cla] * k_shot)
        base_labels_query.extend([cla] * m_query)

    base_labels_support = torch.tensor(base_labels_support)
    base_labels_query = torch.tensor(base_labels_query)
    novel_labels_support = torch.tensor(novel_labels_support)
    novel_labels_query = torch.tensor(novel_labels_query)

    return np.array(base_id_support), np.array(base_id_query), np.array(novel_id_support), np.array(novel_id_query), \
        base_class_selected, novel_class_selected, base_labels_support, base_labels_query, novel_labels_support, novel_labels_query


def val_task_generator(m_query, val_pool, base_id):
    # 这里的val_pool进行session 0 的预测

    base_id_query = []
    base_labels_query = []

    for cla in base_id:
        num_samples = len(val_pool[cla])
        if num_samples < ( m_query):
            print(f"Error: Not enough samples for label {cla}. Required: {m_query}, Available: {num_samples}")

        temp = random.sample(val_pool[cla], m_query)
        base_id_query.extend(temp[:m_query])
        base_labels_query.extend([cla] * m_query)

    base_labels_query = torch.tensor(base_labels_query)

    return np.array(base_id_query), base_labels_query


def incremental_test_task_generator(id_by_class, val_pool, n_way, k_shot, m_query, base_id, novel_id, new_base_id):
    """

    :param id_by_class:
    :param val_pool: id pool for validation
    :param n_way:
    :param k_shot:
    :param m_query:
    :param base_id: novel class in incremental train stage
    :param novel_id: novel class in test stage
    :param new_base_id:
    :return:
    """
    # base_id用来筛选meta_train使用的类别，生成query_embeddings，用于预测
    # new_base_id用于存放出现过的novel_id_test，生成support和query, 用于更新memory和预测
    # novel_id存放未出现过的novel_id_test, 生成support和query, 用于更新memory和预测

    """print('base_id:',len(base_id))
    print('novel_id:',len(novel_id))"""
    # sample class indices
    base_class_selected = base_id + new_base_id
    novel_class_selected = random.sample(novel_id, n_way)

    novel_id_support = []
    novel_id_query = []
    base_id_support = []
    base_id_query = []

    novel_labels_support = []
    novel_labels_query = []
    base_labels_support = []
    base_labels_query = []

    # Sample for novel classes
    for cla in novel_class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        novel_id_support.extend(temp[:k_shot])
        novel_id_query.extend(temp[k_shot:])
        novel_labels_support.extend([cla] * k_shot)
        novel_labels_query.extend([cla] * m_query)

    # Sample for base classes
    # The prototypes of base class are fixed after incremental train
    for cla in base_id:
        temp = random.sample(val_pool[cla], m_query)
        base_id_query.extend(temp[:m_query])
        base_labels_query.extend([cla] * m_query)

    for cla in new_base_id:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        base_id_support.extend(temp[:k_shot])
        base_id_query.extend(temp[k_shot:])
        base_labels_support.extend([cla] * k_shot)
        base_labels_query.extend([cla] * m_query)

    # base_labels_support, base_id_support包括已见过的novel_train_id
    # base_labels_query, base_id_query包括meta_train和已见过的novel_train_id

    base_labels_support = torch.tensor(base_labels_support)
    base_labels_query = torch.tensor(base_labels_query)
    novel_labels_support = torch.tensor(novel_labels_support)
    novel_labels_query = torch.tensor(novel_labels_query)

    return np.array(base_id_support), np.array(base_id_query), np.array(novel_id_support), np.array(novel_id_query), \
        base_class_selected, novel_class_selected, base_labels_support, base_labels_query, novel_labels_support, novel_labels_query


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def NLLLoss(logs, targets, weights=None):
    if weights is not None:
        out = (logs * weights)[range(len(targets)), targets]
    else:
        out = logs[range(len(targets)), targets]
    return -torch.mean(out)


def save_object(obj, filename):
    with open(filename, 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def batched_temperature_scaled_cosine_distance(query_embeddings, sub_prototypes, temperature=0.07, batch_size=25):

    """
    计算带温度的余弦距离
    :param query_embeddings: 查询向量，形状为 [batch_size, embedding_dim]
    :param sub_prototypes: 原型集合，是一个列表，其中每个元素形状为 [embedding_dim]
    :param temperature: 温度缩放因子
    :return: 形状为 [num_prototypes, batch_size] 的距离矩阵
    """
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    sub_prototypes = F.normalize(sub_prototypes, p=2, dim=1)
    
    batched_dists = []
    for i in range(0, len(sub_prototypes), batch_size):
        sub_prototypes_batch = sub_prototypes[i:i+batch_size]
        cosine_sim_batch = torch.stack([F.cosine_similarity(query_embeddings, sub_proto, dim=1) for sub_proto in sub_prototypes_batch])
        scaled_sim_batch = cosine_sim_batch / temperature
        sub_dists_batch = 1 - scaled_sim_batch
        batched_dists.append(sub_dists_batch)
    
    return torch.cat(batched_dists, dim=0)



def save_best_model(encoder, optimizer_encoder, best_epoch, best_model_weights):
    """
    暂存最优模型权重到内存
    """
    best_model_weights['encoder'] = copy.deepcopy(encoder.state_dict())
    best_model_weights['optimizer_encoder'] = copy.deepcopy(optimizer_encoder.state_dict())
    print(f"Best model weights are saved at epoch {best_epoch}.")


def load_best_model(encoder, optimizer_encoder, best_epoch, best_model_weights):
    """
    从内存加载最优模型权重。
    """
    encoder.load_state_dict(best_model_weights['encoder'])
    optimizer_encoder.load_state_dict(best_model_weights['optimizer_encoder'])
    print(f"Best model weights are loaded at epoch {best_epoch}.")

def detach_structure(data):
    """
    遍历嵌套数据结构，递归处理字典、列表和张量。
    - 对字典，逐键递归调用。
    - 对列表，逐项递归调用。
    - 对张量，调用 .detach()。
    - 对其他类型，保持不变。
    """
    if isinstance(data, dict):
        return {k: detach_structure(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [detach_structure(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.detach()
    else:
        return data  # 保持其他类型不变
    
def detach_prototypes(prototypes):
    return {
        lbl: {
            'class_prototype': proto_info['class_prototype'].detach(),
            'sub_prototypes': [sub_proto.detach() for sub_proto in proto_info['sub_prototypes']],
            'densities': proto_info['densities'],  # 标量或列表，无需 detach
        }
        for lbl, proto_info in prototypes.items()
    }


def split_train_val_test(id_by_class, class_id, val_size=10, train_size=15):
    train_pool = {}
    val_pool = {}
    test_pool = {}

    for cla in class_id:
        node_idx = id_by_class[cla]
        total_samples = len(node_idx)
        
        random.shuffle(node_idx)
        # 检查是否有足够的样本进行划分
        if total_samples < 2 * train_size + val_size:
            train_pool[cla] = node_idx[:train_size]  # 前15个样本用于训练
            val_pool[cla] = node_idx[train_size: train_size+val_size]  # 中间部分用于验证
            test_pool[cla] = node_idx[train_size:]  # 剩余部分用于测试
            continue

        # 按照给定的比例和大小进行划分
        train_pool[cla] = node_idx[:train_size]  # 前15个样本用于训练
        val_pool[cla] = node_idx[train_size: train_size+val_size]  # 中间部分用于验证
        test_pool[cla] = node_idx[train_size+val_size:]  # 剩余部分用于测试

    return train_pool, val_pool, test_pool

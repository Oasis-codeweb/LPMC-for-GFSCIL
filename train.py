from __future__ import division
from __future__ import print_function

import random
import time
import argparse
import numpy as np
import os
from copy import deepcopy
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from utils import *
from DBSCAN import Memory
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--way', type=int, default=3, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/cora-full/coauthorCS/Computers')
parser.add_argument('--lambda_mse', required=True, default=1e-5,
                    help='parameter to balance the influence of MSELoss on Loss Function')
parser.add_argument('--pretrain', type=bool, default=True, help='copy the pretrained model parameters or not')
parser.add_argument('--incremental', type=bool, default=False, help='incremental train or not')
parser.add_argument('--pretrained_model', default='GCN', help='GCN/GAT/GraphSAGE')
parser.add_argument('--inner_step', type=int, default='10', help='step to fit')
parser.add_argument('--incre_inner_step', type=int, default='500', help='step to fine-tune')
parser.add_argument('--MinPts', type=int, default='5', help='Minimum points for DBSCAN clustering')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
data, id_by_class, base_id, novel_id, node_list, num_all_nodes = load_raw_data(dataset)

x = data.x.detach()
edge_index = data.edge_index
edge_index = to_undirected(edge_index)
data = Data(x=x, edge_index=edge_index)
data = data.to(device)

"""for cla in novel_id:
    print("class {}: {} nodes".format(cla,len(id_by_class[cla])))

for cla in base_id:
    print("class {}: {} nodes".format(cla,len(id_by_class[cla])))"""

"""cache_path = os.path.join("./cache", (str(args.dataset) + "_" +str(args.pretrained_model) + ".pkl"))
cache = load_object(cache_path)
base_id = cache["base_id"]
novel_train_id = cache["novel_train_id"]
novel_test_id = cache["novel_test_id"]"""

novel_train_id, novel_test_id = split_novel_data(novel_id, dataset)

base_id, novel_train_id, novel_test_id = check_id(id_by_class, base_id, novel_train_id, novel_test_id, args.way)
print('number of base_id:{}, novel_trian_id:{}, novel_test_id:{}'.format(len(base_id), len(novel_train_id),
                                                                         len(novel_test_id)))
# print("novel_test_id:{}", novel_test_id)
# print('base_id:{}, novel_trian_id:{}, novel_test_id:{}'.format(base_id, novel_train_id,novel_test_id))
# Model and optimizer
if args.pretrained_model == 'GCN':
    encoder = GCN(input_dim=data.x.shape[1], hid_dim=args.hidden, out_dim=None, num_layer=2, JK="last",
                  drop_ratio=args.drop_ratio,
                  pool='mean').to(device)
elif args.pretrained_model == 'GAT':
    encoder = GAT(input_dim=data.x.shape[1], hid_dim=args.hidden, out_dim=None, num_layer=2, JK="last",
                  drop_ratio=args.drop_ratio,
                  pool='mean').to(device)
elif args.pretrained_model == 'GraphSAGE':
    encoder = GraphSAGE(input_dim=data.x.shape[1], hid_dim=args.hidden, out_dim=None, num_layer=2, JK="last",
                        drop_ratio=args.drop_ratio,
                        pool='mean').to(device)
else:
    raise ValueError(f"Unsupported encoder type: {args.pretrained_model}")

optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.pretrain:
    encoder.load_state_dict(
        torch.load(f'./pre_trained_gnn/{dataset}/SimGRACE.{args.pretrained_model}.128hidden_dim.pth',
                   map_location='cpu'))
    print("Successfully loaded pre-trained weights!")


def get_embeddings(id_by_class, data, base_id, novel_id):
    embeddings = encoder(data.x, data.edge_index)
    z_dim = embeddings.size()[1]

    # 初始化用于存储指定类别的embeddings和标签
    base_embeddings = []
    base_labels = []
    novel_embeddings = []
    novel_labels = []

    # 遍历class_id，收集对应的embeddings和标签
    for cla in base_id:
        if cla in id_by_class:
            cla_embeddings = embeddings[id_by_class[cla]]
            base_embeddings.append(cla_embeddings)
            base_labels.append(torch.full((len(cla_embeddings),), cla))
        else:
            raise KeyError(cla)

    for cla in novel_id:
        if cla in id_by_class:
            cla_embeddings = embeddings[id_by_class[cla]]
            base_embeddings.append(cla_embeddings)
            base_labels.append(torch.full((len(cla_embeddings),), cla))
        else:
            raise KeyError(cla)

    # 将列表转换为张量
    base_embeddings = torch.cat(base_embeddings)
    base_labels = torch.cat(base_labels)

    novel_embeddings = torch.cat(novel_embeddings)
    novel_labels = torch.cat(novel_labels)

    return base_embeddings, base_labels, novel_embeddings, novel_labels


def classification(query_embeddings, query_labels, prototypes):
    """# 创建标签映射表，将 query_labels 标签映射到 prototypes 的索引范围
    label_to_idx = {cls_label: idx for idx, (cls_label, proto_info) in enumerate(prototypes.items())}

    # 将 query_labels 映射到新的索引范围内
    labels_new = torch.tensor([label_to_idx.get(label.item(), -1) for label in query_labels])"""

    # 创建标签映射表，将 query_labels 标签映射到 prototypes 的索引范围
    label_to_idx = {cls_label: idx for idx, (cls_label, proto_info) in enumerate(prototypes.items())}

    """# 打印映射表以进行检查
    print("Label to index mapping:")
    for label, idx in label_to_idx.items():
        print(f"Label: {label}, Index: {idx}")"""

    # 将 query_labels 映射到新的索引范围内，并检查未定义标签
    labels_new = []
    for label in query_labels:
        label_value = label.item()  # 获取具体标签值
        if label_value not in label_to_idx:
            # 如果标签未定义，抛出异常
            raise ValueError(f"Label '{label_value}' is not defined in the prototypes mapping.")
        labels_new.append(label_to_idx[label_value])

    # 转换为 PyTorch 张量
    labels_new = torch.tensor(labels_new)

    num_classes = len(prototypes)
    dists = torch.zeros(query_embeddings.size(0), num_classes)

    # 遍历每个类别的亚原型
    for idx, (cls_label, proto_info) in enumerate(prototypes.items()):
        sub_prototypes = proto_info['sub_prototypes']
        densities = proto_info['densities']

        # query_embeddings的形状是 N X D， 由N个query_embedding构成
        # 使用带温度的余弦距离计算每个query_embedding与每个类的若干个亚原型的平均距离

        # 将 sub_prototypes 列表转换为张量
        sub_prototypes = torch.stack(sub_prototypes).to(device)
        sub_dists = batched_temperature_scaled_cosine_distance(query_embeddings, sub_prototypes)

        # 将 densities 转换为 torch 张量，并扩展其形状为 (2, 1)
        weights = torch.tensor(densities).unsqueeze(1)

        weights = weights.expand_as(sub_dists).to(device)
        weighted_sub_dists = sub_dists * weights
        weighted_avg_dist = weighted_sub_dists.sum(dim=0) / weights.sum(dim=0)

        # avg_dist = sub_dists.mean(dim=0)
        dists[:, idx] = weighted_avg_dist

    # 使用 log_softmax 获取输出
    output = F.log_softmax(-dists, dim=1)

    # 计算分类损失
    loss = NLLLoss(output, labels_new)

    # 计算分类准确率
    acc = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return loss, acc, f1_train


def distil(previous_prototypes, current_prototypes, base_class_selected, temperature=1.0):
    prev_embeddings = []
    curr_embeddings = []

    for label in base_class_selected:
        prev_embeddings.append(previous_prototypes[label]['class_prototype'])
        curr_embeddings.append(current_prototypes[label]['class_prototype'])

    # 将列表转换为张量
    prev_embeddings_tensor = torch.stack(prev_embeddings)
    curr_embeddings_tensor = torch.stack(curr_embeddings)

    """print(f"prev_embeddings_tensor size:{prev_embeddings_tensor.size()}")
    print(f"curr_embeddings_tensor size:{curr_embeddings_tensor.size()}")"""

    if prev_embeddings_tensor.size(1) != curr_embeddings_tensor.size(1):
        print("Warning: The size of old_prototypes and new_prototypes does not match.")
        cosine_loss = torch.tensor(0.0)  # 如果大小不匹配，返回0损失
    else:
        # 对嵌入进行归一化
        prev_embeddings_tensor = F.normalize(prev_embeddings_tensor, p=2, dim=1)
        curr_embeddings_tensor = F.normalize(curr_embeddings_tensor, p=2, dim=1)

        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(prev_embeddings_tensor, curr_embeddings_tensor, dim=1)

        # 使用温度缩放余弦相似度
        scaled_cosine_sim = cosine_sim / temperature

        # 计算损失
        cosine_loss = 1 - scaled_cosine_sim.mean()  # 取平均损失

    return cosine_loss


def train(data, id_support, id_query, labels_support, labels_query, memory):
    encoder.train()
    loss_train = 0.0

    embeddings = encoder(data.x, data.edge_index)

    support_embeddings = embeddings[id_support]
    query_embeddings = embeddings[id_query]

    if args.cuda:
        support_embeddings = support_embeddings.cuda()
        query_embeddings = query_embeddings.cuda()

    # 在memory中，新类创建微簇中心，旧类更新微簇中心
    prototypes = memory(base_embeds=None, base_lbs=None, novel_embeds=support_embeddings,
                        novel_lbs=labels_support)

    # 分类loss的计算
    classification_loss, classification_acc, f1_train = classification(query_embeddings, labels_query, prototypes)

    loss_train = classification_loss
    optimizer_encoder.zero_grad()
    loss_train.backward()
    optimizer_encoder.step()

    return classification_acc, f1_train, memory


def incremental_train(data, base_id_support, base_id_query, novel_id_support, novel_id_query,
                      base_labels_support, base_labels_query, novel_labels_support, novel_labels_query,
                      base_class_selected, novel_class_selected, n_way, k_shot, id_by_class, temp_memory):
    encoder.train()
    loss_train = 0.0

    """
    在memory的外部要干的几件事：1. copy memory 
    * 提取memory中存在类别的若干个微簇中心：M * N
    2. 提取基类节点特征
    3. 提取新类节点特征
    4. 将节点特征输入memory 
    """

    # 计算prototype_embeddings的方法变为根据linear_sum/N来计算，符合流式数据的计算特点
    # train_memory.mc储存微簇中心的统计信息（LS,SS,N），train_memory.prototypes储存微簇中心, hint: 一个类可能有若干个微簇

    # 获取前一个session的微簇聚类信息
    train_memory = Memory(minPts)

    """for k, v in memory.mc.items():
        for vi in v:
            print(f"Type of num_nodes in class {k}: {type(vi['num_nodes'])}")
            print(f"Type of radius in class {k}: {type(vi['radius'])}")"""

    train_memory.mc = detach_structure(temp_memory.mc)
    train_memory.prototypes = detach_prototypes(temp_memory.prototypes)

    id_by_class = {int(k): v for k, v in id_by_class.items()}

    # 提取节点特征
    embeddings = encoder(data.x, data.edge_index)
    z_dim = embeddings.size()[1]

    base_support_embeddings = embeddings[base_id_support]
    novel_support_embeddings = embeddings[novel_id_support]
    base_query_embeddings = embeddings[base_id_query]
    novel_query_embeddings = embeddings[novel_id_query]

    if args.cuda:
        base_support_embeddings = base_support_embeddings.cuda()
        base_query_embeddings = base_query_embeddings.cuda()

        novel_support_embeddings = novel_support_embeddings.cuda()
        novel_query_embeddings = novel_query_embeddings.cuda()

    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    query_labels = torch.cat((base_labels_query, novel_labels_query), dim=0)

    # 在memory中，新类创建微簇中心，旧类更新微簇中心
    prototypes = train_memory(base_support_embeddings, base_labels_support, novel_support_embeddings,
                              novel_labels_support)
    """for lbl, proto_info in prototypes.items():
        print(f"Prototype - Label {lbl}:",
            proto_info['class_prototype'].requires_grad,
            [sp.requires_grad for sp in proto_info['sub_prototypes']])"""
    """
    经过memory之后，得到了每个类的sub_prototype, prototype
    1. sub_prototype作为亚原型参与计算分类loss
    2. class_prototype作为类原型，参与计算蒸馏loss
    """

    # 分类loss的计算
    classification_loss, classification_acc, f1_train = classification(query_embeddings, query_labels, prototypes)

    # 蒸馏loss的计算
    previous_prototypes = detach_prototypes(temp_memory.prototypes)
    current_prototypes = prototypes

    # Ensure all prototypes contain 'class_prototype' for distillation
    flag = all(
        'class_prototype' in proto_info and proto_info['class_prototype'].isfinite().all()
        for proto_info in prototypes.values()
    )

    distil_loss = torch.tensor(0.0)
    if flag:
        distil_loss = distil(previous_prototypes, current_prototypes, base_class_selected)
        distil_loss = torch.tensor(lambda_mse).float() * distil_loss

    loss_train = classification_loss + distil_loss
    # print('classification_loss, distil_loss: ', classification_loss, distil_loss)
    # print("loss_train:", loss_train)

    optimizer_encoder.zero_grad()
    loss_train.backward()
    optimizer_encoder.step()

    # update memory
    temp_memory.mc = detach_structure(train_memory.mc)
    temp_memory.prototypes = detach_prototypes(train_memory.prototypes)

    # temp_memory.cluster_access_count = train_memory.cluster_access_count.copy()

    return classification_acc, f1_train, temp_memory


"""
    在test/validation部分干的事：
    1. 不再修改memory内部原有的内容，只能增加新类微簇中心的统计信息（也就是冻结encoder权重）
    2. 提取新类支持集节点特征
    3. 将节点特征输入memory，将节点特征输入memory
    4. 计算准确率（提取查询集特征）
"""


def incremental_val(data, base_id_query, base_labels_query, val_memory):
    encoder.eval()

    """val_memory = Memory(minPts)
    # update memory
    val_memory.mc = {k: [dict(vi,
                              center=vi['center'].detach(),
                              linear_sum=vi['linear_sum'].detach(),
                              square_sum=vi['square_sum'].detach(),
                              num_nodes=vi['num_nodes'],  # num_nodes is a scalar
                              radius=vi['radius'].detach(),
                              has_new_samples=vi['has_new_samples'])
                         for vi in v]
                     for k, v in memory.mc.items()}
    val_memory.prototypes = {k: dict(v, class_prototype=v['class_prototype'].detach(),
                                     sub_prototypes=[sub.detach() for sub in v['sub_prototypes']]) for k, v in
                             memory.prototypes.items()}
    val_memory.cluster_access_count = memory.cluster_access_count.copy()"""

    # 提取节点特征
    embeddings = encoder(data.x, data.edge_index)
    z_dim = embeddings.size()[1]

    query_embeddings = embeddings[base_id_query]
    query_labels = base_labels_query
    # query_labels = torch.cat(base_labels_query, dim=0)
    prototypes = val_memory.prototypes
    if args.cuda:
        query_embeddings = query_embeddings.cuda()

    # 分类loss的计算
    _, acc_val, f1_val = classification(query_embeddings, query_labels, prototypes)

    return acc_val, f1_val


def incremental_test(data, base_id_support, base_id_query, novel_id_support, novel_id_query,
                     base_labels_support, base_labels_query, novel_labels_support, novel_labels_query,
                     id_by_class, temp_memory, fund_lbl, incre_lbl):
    encoder.eval()

    id_by_class = {int(k): v for k, v in id_by_class.items()}

    # 提取节点特征
    embeddings = encoder(data.x, data.edge_index)
    z_dim = embeddings.size()[1]

    base_support_embeddings = embeddings[base_id_support]
    novel_support_embeddings = embeddings[novel_id_support]
    base_query_embeddings = embeddings[base_id_query]
    novel_query_embeddings = embeddings[novel_id_query]

    if args.cuda:
        base_support_embeddings = base_support_embeddings.cuda()
        base_query_embeddings = base_query_embeddings.cuda()

        novel_support_embeddings = novel_support_embeddings.cuda()
        novel_query_embeddings = novel_query_embeddings.cuda()

    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    # print("query embeddings size:{}".format(query_embeddings.size()))
    query_labels = torch.cat((base_labels_query, novel_labels_query), dim=0)

    # 在memory中，新类创建微簇中心，旧类更新微簇中心
    prototypes = temp_memory(base_support_embeddings, base_labels_support, novel_support_embeddings, novel_labels_support)
    #prototypes = temp_memory(base_embeds = None, base_lbs = None, novel_embeds = base_support_embeddings, novel_lbs=base_labels_support)
    number_of_keys = len(prototypes)

    """# 打印键的数量
    print("Number of keys in prototype dictionary:", number_of_keys)
    print("Prototypes keys:", prototypes.keys())
    print("Unique query labels:", set(label.item() for label in query_labels))"""

    # 分类loss的计算
    _, acc_test, f1_test = classification(query_embeddings, query_labels, prototypes)

    fundamental_id_query = []
    fundamental_labels_query = []
    incremental_id_query = []
    incremental_labels_query = []

    # 筛选出属于 fundamental_label 的 query_labels
    for i, label in enumerate(base_labels_query):
        if label in fund_lbl:
            fundamental_id_query.append(base_id_query[i])
            fundamental_labels_query.append(label)
        else:
            incremental_id_query.append(base_id_query[i])
            incremental_labels_query.append(label)

    for i, label in enumerate(novel_labels_query):
        incremental_id_query.append(novel_id_query[i])
        incremental_labels_query.append(label)

    fundamental_query_embeddings = embeddings[fundamental_id_query]
    incremental_query_embeddings = embeddings[incremental_id_query]
    if args.cuda:
        fundamental_query_embeddings = fundamental_query_embeddings.cuda()
        incremental_query_embeddings = incremental_query_embeddings.cuda()

    print("fundamental embedding size:{},incremental embedding size:{}".format(fundamental_query_embeddings.size(),
                                                                                incremental_query_embeddings.size()))

    fundamental_labels_query = torch.tensor(fundamental_labels_query)
    incremental_labels_query = torch.tensor(incremental_labels_query)

    """# bs_SA_id 和 nv_SA_id 是包含相应键的列表
    fund_prototypes = {label: prototypes[label] for label in fund_lbl}
    incre_prototypes = {label: prototypes[label] for label in incre_lbl}"""

    # 接下来使用提取出的原型进行分类计算
    _, fund_acc_test, base_f1_test = classification(fundamental_query_embeddings, fundamental_labels_query,
                                                    prototypes)
    _, incre_acc_test, novel_f1_test = classification(incremental_query_embeddings, incremental_labels_query,
                                                        prototypes)

    return acc_test, fund_acc_test, incre_acc_test, f1_test, temp_memory



if __name__ == '__main__':

    """# 检查数据集的构成
    total_nodes = 0
    print(f"Number of classes in id_by_class: {len(id_by_class)}")
    for cla in id_by_class:
        node_idx = id_by_class[cla]
        print(f"Class {cla}: {len(node_idx)} nodes")
        total_nodes += len(node_idx)  # 累加每个类的节点数量

    print(f"Total number of nodes across all classes: {total_nodes}")
    print(f"novel train class: {novel_train_id}")
    print(f"novel test class: {novel_test_id}")"""

    # 按照类别数量分割数据集成三个部分：预训练(base_id), 训练 (novel_train_id)，测试 (novel_test_id)
    # pretrain使用分类任务预训练模型
    # meta-train使用伪增量学习
    # test进行类增量分类任务

    n_way = args.way
    k_shot = args.shot
    m_query = args.qry
    inner_step = args.inner_step
    meta_val_num = 5
    meta_test_num = 10
    pretrain_test_num = 10
    lambda_mse = args.lambda_mse
    lambda_mse = float(lambda_mse)

    minPts = args.MinPts
    memory = Memory(minPts)
    temp_memory = Memory(minPts)
    # Train model
    t_total = time.time()
    meta_train_acc = []
    # Initialize variables to store the best accuracy and F1 score
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_epoch = 0
    best_avg_acc = 0.0
    best_val_acc = 0.0
    max_SA = 0.0
    best_meta_test_acc = []
    best_meta_test_f1 = []
    best_val_f1 = 0

    """# 将novel_train_id分成meta_train和meta_test, meta_train 用于训练，meta_test 用于val
    novel_train_pool, novel_val_pool = split_novel_train_val(id_by_class, novel_train_id)
    base_train_pool, base_val_pool = split_novel_train_val(id_by_class, base_id)"""

    train_id = novel_train_id + base_id
    meta_train_pool, meta_val_pool = split_train_val(id_by_class, train_id)
    incre_train_pool, incre_val_pool, incre_test_pool = split_train_val_test(id_by_class, novel_test_id)


    all_base_class_selected = []
    novel_class_left = deepcopy(novel_train_id)
    base_class = deepcopy(base_id)
    all_base_class_selected.extend(base_class)
    flag_incremental = args.incremental
    
    novel_data_pool = []
    random.shuffle(novel_test_id) 
    # 填充 novel_data_pool, 固定模型可获取的新类知识，避免数据泄露
    for cla in novel_test_id:
        # 采样得到当前类别的支持集和查询集（训练集）
        temp_train = random.sample(incre_train_pool[cla], k_shot + m_query)
        current_group_train = {
            'novel_id_support': temp_train[:k_shot],
            'novel_id_query': temp_train[k_shot:],
            'novel_labels_support': [cla] * k_shot,
            'novel_labels_query': [cla] * m_query
        }
        novel_data_pool.append(current_group_train)

    for episode in range(args.episodes):
        number_of_keys = len(memory.prototypes)
        # 打印键的数量
        print("Number of keys in memory:", number_of_keys)

        number_of_keys = len(temp_memory.prototypes)
        # 打印键的数量
        print("Number of keys in temp_memory:", number_of_keys)

        for i in range(inner_step):
            if i < inner_step -2 :
                # reset memory
                temp_memory.mc = detach_structure(memory.mc)
                temp_memory.prototypes = detach_prototypes(memory.prototypes)
            else:
                # copy memory
                temp_memory.mc = detach_structure(temp_memory.mc)
                temp_memory.prototypes = detach_prototypes(temp_memory.prototypes)

            if all_base_class_selected == []:
                if n_way == 1:
                    # To prevent meaningless training
                    class_selected = random.sample(novel_class_left, n_way + 1)
                    # print(f"all_base_class_selected is empty! class selected:{class_selected}")
                else:
                    class_selected = random.sample(novel_class_left, n_way)

                id_support, id_query, labels_support, labels_query, base_class_selected = \
                    task_generator(class_selected, meta_train_pool, n_way, k_shot, m_query)

                acc_train, f1_train, temp_memory = train(data, id_support, id_query, \
                                                         labels_support, labels_query, temp_memory)

                if i == inner_step - 1:
                    flag_incremental = True
                    if n_way == 1:
                        novel_class_selected = random.sample(class_selected, n_way)
                    else:
                        novel_class_selected = class_selected

                """print("Debugging temp_memory.mc:")
                print(temp_memory.mc)"""

                """print("Checking temp_memory.mc contents and types:")

                # 遍历 temp_memory.mc 字典
                for key, value_list in temp_memory.mc.items():
                    print(f"Key: {key}")
                    # 遍历每个列表中的字典
                    for value_dict in value_list:
                        print("  Value dict:")
                        for sub_key, sub_value in value_dict.items():
                            print(f"    {sub_key}: Type: {type(sub_value)}, Content: {sub_value}")"""

            else:
                if flag_incremental:
                    # meta-train
                    base_id_support, base_id_query, novel_id_support, novel_id_query, base_class_selected, novel_class_selected, \
                        base_labels_support, base_labels_query, novel_labels_support, novel_labels_query = \
                        incremental_train_task_generator(id_by_class, meta_train_pool, n_way, k_shot, m_query,
                                                         all_base_class_selected,
                                                         novel_class_left, base_class)

                    acc_train, f1_train, temp_memory = incremental_train(data, base_id_support, base_id_query,
                                                                         novel_id_support,
                                                                         novel_id_query,
                                                                         base_labels_support, base_labels_query,
                                                                         novel_labels_support, novel_labels_query,
                                                                         base_class_selected, novel_class_selected,
                                                                         n_way, k_shot,
                                                                         id_by_class, temp_memory
                                                                         )

                else:
                    # temp_memory.reset_memory()
                    # from novel_train_id extract novel classes, and then classify
                    id_support, id_query, labels_support, labels_query, base_class_selected = \
                        task_generator(base_class, meta_train_pool, n_way, k_shot, m_query)

                    acc_train, f1_train, temp_memory = train(data, id_support, id_query, \
                                                             labels_support, labels_query, temp_memory)
                    novel_class_selected = []
                    if i == inner_step - 1:
                        flag_incremental = True

            # print(f"Episode: {episode}, Sample: {i}, Train Accuracy: {acc_train}, F1 Score: {f1_train}")

        # print(f"epoch {episode}, novel class selected:{novel_class_selected}")
        all_base_class_selected.extend(novel_class_selected)
        novel_class_left = list(set(novel_class_left) - set(novel_class_selected))

        # update memory
        memory.mc = detach_structure(temp_memory.mc)
        memory.prototypes = detach_prototypes(temp_memory.prototypes)

        # number_of_keys = len(memory.prototypes)
        # 打印键的数量
        # print("Number of keys in prototype dictionary:", number_of_keys)

        meta_train_acc.append(acc_train)
        print(f"Meta train Epoch: {episode}, Current_Train_Accuracy: {acc_train}")
        # print(f"Meta-Train_Accuracy (Average up to this point): {np.array(meta_train_acc).mean(axis=0)}")

        if len(novel_class_left) < n_way:
            print("---------------------Validation at Episode {}---------------------------".format(episode))
            # Sampling a pool of tasks for validation
            val_task_pool = [
                val_task_generator(m_query, meta_val_pool, all_base_class_selected) for i
                in range(meta_val_num)]

            # validation
            meta_val_acc = []
            meta_val_f1 = []

            with torch.no_grad():
                for idx in range(meta_val_num):
                    base_id_query, base_labels_query = val_task_pool[idx]

                    acc_val, f1_val = incremental_val(data, base_id_query, base_labels_query, memory)
                    meta_val_acc.append(acc_val)
                    meta_val_f1.append(f1_val)
                    mean_val_acc = np.array(meta_val_acc).mean(axis=0)
                    mean_val_f1 = np.array(meta_val_f1).mean(axis=0)

            print("Meta val_Accuracy: {}, Meta val_F1: {}".format(mean_val_acc, mean_val_f1))

            print("Model is initialized with meta-training! Class incremental learning starts!")

            meta_test_acc = []
            meta_test_f1 = []

            all_base_class_selected_incre = deepcopy(train_id)
            novel_class_left_incre = deepcopy(novel_test_id)

            all_base_class_selected_test = deepcopy(train_id)
            novel_class_left_test = deepcopy(novel_test_id)
            new_base_class = []

            old_SA_id = deepcopy(train_id)
            new_SA_id = []
            session_number = 0

            incre_memory = Memory(minPts)
            incre_memory.mc = detach_structure(memory.mc)
            incre_memory.prototypes = detach_prototypes(memory.prototypes)

            incre_temp_memory = Memory(minPts)
            incre_temp_memory.mc = detach_structure(memory.mc)
            incre_temp_memory.prototypes = detach_prototypes(memory.prototypes)

            novel_test_pool = []
            combined_val_pool = {**meta_val_pool, **incre_val_pool}
            combined_test_pool = {**meta_val_pool, **incre_test_pool}

            """print("\nmeta_val_pool labels and corresponding ID counts:")
            for label, ids in meta_val_pool.items():
                print(f"Label: {label}, Number of IDs: {len(ids)}")

            print("\nincre_val_pool labels and corresponding ID counts:")
            for label, ids in incre_val_pool.items():
                print(f"Label: {label}, Number of IDs: {len(ids)}")

            print("\ncombined_val_pool labels and corresponding ID counts:")
            for label, ids in combined_val_pool.items():
                print(f"Label: {label}, Number of IDs: {len(ids)}")"""
            


            base_id_support_incre, base_id_query_incre, base_labels_support_incre, base_labels_query_incre, base_class_selected_incre = \
                task_generator(train_id, meta_train_pool, n_way, k_shot, m_query)

            for idx in range(meta_test_num):

                # incremental fine-tune
                print("---------------------Incremental learning at Episode {}---------------------------".format(episode))

                best_val_acc_incre = 0

                novel_id_support_incre = []
                novel_id_query_incre = []
                novel_labels_support_incre = []
                novel_labels_query_incre = []

                novel_data_group = novel_data_pool[idx * n_way: (idx + 1) * n_way]
                novel_class_selected_incre = []

                for novel_data in novel_data_group:
                    novel_class_selected_incre.append(novel_data['novel_labels_support'][0])
                    novel_id_support_incre.extend(novel_data['novel_id_support'])
                    novel_id_query_incre.extend(novel_data['novel_id_query'])
                    novel_labels_support_incre.extend(novel_data['novel_labels_support'])
                    novel_labels_query_incre.extend(novel_data['novel_labels_query'])

                novel_id_support_incre = np.array(novel_id_support_incre)
                novel_id_query_incre = np.array(novel_id_query_incre)
                novel_labels_support_incre = torch.tensor(novel_labels_support_incre)
                novel_labels_query_incre = torch.tensor(novel_labels_query_incre)

                print(f"novel_class_selected in incremental learning is {novel_class_selected_incre}")
                all_base_class_selected_incre.extend(novel_class_selected_incre)

                # fine-tune
                for i in range(args.incre_inner_step):
                    acc_train, f1_train, incre_temp_memory = incremental_train(data, base_id_support_incre, base_id_query_incre,
                                                                         novel_id_support_incre,
                                                                         novel_id_query_incre,
                                                                         base_labels_support_incre, base_labels_query_incre,
                                                                         novel_labels_support_incre, novel_labels_query_incre,
                                                                         base_class_selected_incre, novel_class_selected,
                                                                         n_way, k_shot,
                                                                         id_by_class, incre_temp_memory
                                                                         )

                    if i % 10 == 9:
                        val_task_pool = [
                            val_task_generator(m_query, combined_val_pool, all_base_class_selected_incre) for i
                            in range(5)]

                        # validation
                        meta_val_acc_incre = []
                        meta_val_f1_incre = []

                        with torch.no_grad():
                            for val_idx in range(5):
                                id_query_incre, labels_query_incre = val_task_pool[val_idx]

                                acc_val, f1_val = incremental_val(data, id_query_incre, labels_query_incre, incre_temp_memory)
                                meta_val_acc_incre.append(acc_val)
                                meta_val_f1_incre.append(f1_val)
                                mean_val_acc_incre = np.array(meta_val_acc_incre).mean(axis=0)
                                mean_val_f1_incre = np.array(meta_val_f1_incre).mean(axis=0)

                        print("Meta val_Accuracy: {}, Meta val_F1: {}".format(mean_val_acc_incre, mean_val_f1_incre))

                        file_name = "{}.{}.{}way.{}shot.session{}.MinPts{}.pth".format(dataset, args.pretrained_model, args.way,
                                                                              args.shot, idx + 1, args.MinPts)
                        file_path = os.path.join("./model_checkpoint", dataset, args.pretrained_model, file_name)

                        # 检查目录是否存在，如果不存在则创建
                        if not os.path.exists(os.path.dirname(file_path)):
                            os.makedirs(os.path.dirname(file_path))

                        if mean_val_acc_incre > best_val_acc_incre:
                            best_val_acc_incre = mean_val_acc_incre
                            # save model
                            torch.save(encoder.state_dict(), file_path)
                            # save memory
                            incre_temp_memory.mc = detach_structure(incre_temp_memory.mc)
                            incre_temp_memory.prototypes = detach_prototypes(incre_temp_memory.prototypes)
                            # incre_memory备份当前表现最好的memory
                            incre_memory.mc = detach_structure(incre_temp_memory.mc)
                            incre_memory.prototypes = detach_prototypes(incre_temp_memory.prototypes)
                            print(f"session {idx+1} , Model is saved")
                        else:
                            # reset memory
                            incre_temp_memory.mc = detach_structure(incre_memory.mc)
                            incre_temp_memory.prototypes = detach_prototypes(incre_memory.prototypes)
                    else:
                        # reset memory
                        incre_temp_memory.mc = detach_structure(incre_memory.mc)
                        incre_temp_memory.prototypes = detach_prototypes(incre_memory.prototypes)

                # update base data
                base_id_support_incre = np.concatenate((base_id_support_incre, novel_id_support_incre))
                base_id_query_incre = np.concatenate((base_id_query_incre, novel_id_query_incre))
                base_labels_support_incre = torch.cat((base_labels_support_incre, novel_labels_support_incre), dim=0)
                base_labels_query_incre = torch.cat((base_labels_query_incre, novel_labels_query_incre), dim=0)

                # test
                print("---------------------Test at Episode {}---------------------------".format(episode))

                novel_id_support_test = []
                novel_id_query_test = []
                novel_labels_support_test = []
                novel_labels_query_test = []

                test_memory = Memory(minPts)
                test_memory.mc = detach_structure(incre_memory.mc)
                test_memory.prototypes = detach_prototypes(incre_memory.prototypes)

                number_of_keys = len(test_memory.prototypes)
                # 打印键的数量
                print("Number of keys in memory:", number_of_keys)

                # 填充novel_test_pool
                for cla in novel_test_id:
                    # 采样得到当前类别的测试集
                    temp_test = random.sample(incre_test_pool[cla], k_shot + m_query)
                    current_group_test = {
                        'novel_id_support': temp_test[:k_shot],
                        'novel_id_query': temp_test[k_shot:],
                        'novel_labels_support': [cla] * k_shot,
                        'novel_labels_query': [cla] * m_query
                    }
                    novel_test_pool.append(current_group_test)

                if os.path.isfile(file_path):  
                    print(f"Loading model weights from {file_path}")
                    checkpoint = torch.load(file_path, map_location=device)
                    encoder.load_state_dict(checkpoint)  

                with (torch.no_grad()):
                    session_number = session_number + 1
                    # print("number of novel_class_left:", len(novel_class_left_test))
                    # print(f"all_base_class_selected_test:{all_base_class_selected_test}")
                    base_id_support_test, base_id_query_test, base_labels_support_test, base_labels_query_test, base_class_selected_test = \
                        task_generator(all_base_class_selected_test, combined_test_pool, n_way, k_shot, m_query)


                    novel_test_group = novel_test_pool[idx * n_way: (idx + 1) * n_way]
                    novel_class_selected_test = []

                    for novel_data in novel_test_group:
                        novel_class_selected_test.append(novel_data['novel_labels_support'][0])
                        novel_id_support_test.extend(novel_data['novel_id_support'])
                        novel_id_query_test.extend(novel_data['novel_id_query'])
                        novel_labels_support_test.extend(novel_data['novel_labels_support'])
                        novel_labels_query_test.extend(novel_data['novel_labels_query'])
                    
                    novel_id_support_test = np.array(novel_id_support_test)
                    novel_id_query_test = np.array(novel_id_query_test)
                    novel_labels_support_test = torch.tensor(novel_labels_support_test)
                    novel_labels_query_test = torch.tensor(novel_labels_query_test)

                    new_SA_id.extend(novel_class_selected_test)
                    print(f"novel_class_selected in test is {novel_class_selected_test}")

                    # if session_number == len(novel_test_id) // n_way:
                    acc_test, fund_acc_test, incre_acc_test, f1_test, test_memory = \
                        incremental_test(data, base_id_support_test, base_id_query_test,
                                            novel_id_support_test,
                                            novel_id_query_test, base_labels_support_test,
                                            base_labels_query_test,
                                            novel_labels_support_test, novel_labels_query_test,
                                            id_by_class, test_memory,
                                            old_SA_id, new_SA_id)
                    print("base class accuracy:{}, novel class accuracy:{}".format(fund_acc_test, incre_acc_test))
                    """else:
                        acc_test, f1_test, test_memory = \
                            incremental_test(data, base_id_support_test, base_id_query_test,
                                             novel_id_support_test,
                                             novel_id_query_test, base_labels_support_test,
                                             base_labels_query_test,
                                             novel_labels_support_test, novel_labels_query_test,
                                             id_by_class, test_memory,
                                             old_SA_id, new_SA_id)"""

                    new_base_class.extend(novel_class_selected_test)
                    all_base_class_selected_test.extend(novel_class_selected_test)
                    novel_class_left_test = list(set(novel_class_left_test) - set(novel_class_selected_test))

                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)
                    print("Meta test_Accuracy: {}, Meta test_F1: {}, Session:{}".format(np.array(meta_test_acc)[-1],
                                                                                        np.array(meta_test_f1)[-1],
                                                                                        idx + 1))
                    if len(novel_class_left_test) < n_way:
                        break

            # The last session
            SA = (fund_acc_test + incre_acc_test) / session_number

            # Update the best test results if the current results are better
            if SA > max_SA:
                max_SA = SA
                best_val_acc = mean_val_acc
                best_val_f1 = mean_val_f1
                best_epoch = episode
                best_meta_test_acc = meta_test_acc
                best_meta_test_f1 = meta_test_f1


            all_base_class_selected = []
            all_base_class_selected.extend(base_id)
            novel_class_left = deepcopy(novel_train_id)

    print("---------------------Best Result at Episode {}---------------------------".format(best_epoch))

    print(f"max SA:{max_SA}")
    print("Meta test_Accuracy: {}, Meta test_F1: {}, Session:{}".format(best_val_acc, best_val_f1, 0))

    for idx in range(len(best_meta_test_acc)):
        print("Meta test_Accuracy: {}, Meta test_F1: {}, Session:{}".format(np.array(best_meta_test_acc)[idx],
                                                                            np.array(best_meta_test_f1)[idx],
                                                                            idx + 1))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(
        "DATASET: {}, way: {}, shot: {}, qry: {}, lambda_mse: {}, inner_step:{}, pretrained_model:{}, MinPts:{}".format(
            dataset,
            n_way,
            k_shot,
            m_query,
            lambda_mse,
            inner_step,
            args.pretrained_model,
            minPts))

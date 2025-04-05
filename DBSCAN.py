import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from sklearn.cluster import DBSCAN  # CPU 支持的 DBSCAN
from sklearn.preprocessing import StandardScaler


class Memory(nn.Module):
    def __init__(self, minPts):
        super(Memory, self).__init__()
        self.minPts = minPts  # DBSCAN的最小样本数
        self.mc = {}  # 存储每个类别的微簇信息
        self.prototypes = {}  # 存储每个类别的原型

    def forward(self, base_embeds, base_lbs, novel_embeds, novel_lbs):
        """
        输入:
        - base_embeds: 基础类别的嵌入向量 (Tensor or None)
        - base_lbs: 基础类别的标签 (Tensor or None)
        - novel_embeds: 新类别的嵌入向量 (Tensor)
        - novel_lbs: 新类别的标签 (Tensor)

        输出:
        - self.mc: 存储新类别的微簇信息
        """
        # 检查输入数据是否为空
        if base_embeds is None and base_lbs is None:
            self.update_mc_layer(novel_embeds, novel_lbs)
            return self.update_prototypes()  # 更新原型
        else:
            # 将基础类别的嵌入和标签与新颖类别的嵌入和标签合并
            all_embeds = torch.cat([base_embeds, novel_embeds], dim=0)
            all_lbs = torch.cat([base_lbs, novel_lbs], dim=0)

            self.update_mc_layer(all_embeds, all_lbs)
            return self.update_prototypes()  # 更新原型

    def initialize_microcluster(self, novel_embeds, novel_lbs):
        """
        初始化微簇信息。

        输入:
        - novel_embeds: 新类别的嵌入向量 (Tensor)
        - novel_lbs: 新类别的标签 (Tensor)

        输出:
        - self.mc: 存储新类别的微簇信息
        """
        # 对嵌入进行归一化
        novel_embeds_norm = novel_embeds / novel_embeds.norm(dim=1, keepdim=True)

        # 按类别分组
        unique_labels = novel_lbs.unique()  # 在 PyTorch 中获取唯一标签
        # (f"unique labels:{unique_labels}")
        for label in unique_labels:
            # 筛选出当前类别的嵌入
            class_indices = (novel_lbs == label).nonzero(as_tuple=True)[0]
            class_embeds_norm = novel_embeds_norm[class_indices]
            class_embeds = novel_embeds[class_indices]
            # print("unique label {} has {} class embeddings!".format(label, len(class_embeds)))

            # 使用 DBSCAN 聚类
            dbscan = DBSCAN(eps=0.5, min_samples=self.minPts, metric='cosine')
            cluster_labels = dbscan.fit_predict(class_embeds_norm.cpu().detach().numpy())

            # 提取微簇信息
            micro_clusters = {}
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:
                    # 将每个噪声点初始化为独立的微簇
                    noise_indices = np.where(cluster_labels == -1)[0]
                    for idx in noise_indices:
                        point = class_embeds[idx].unsqueeze(0)  # 将单个点变为 1xD 的张量
                        micro_clusters[idx] = {
                            'center': point.squeeze(0),  # 噪声点的中心就是它本身
                            'linear_sum': point.squeeze(0),
                            'square_sum': (point ** 2).squeeze(0),
                            'num_nodes': 1,  # 每个噪声点作为独立微簇只有 1 个节点
                            'radius': 1e-5,  # 初始化为很小的半径
                        }
                        continue
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_points = class_embeds[cluster_indices]

                # 计算微簇中心、线性和、平方和、节点数和半径
                center = cluster_points.mean(dim=0)
                linear_sum = cluster_points.sum(dim=0)
                square_sum = (cluster_points ** 2).sum(dim=0)
                num_nodes = len(cluster_indices)
                radius = self.update_radius_on_stats({
                    'center': center,
                    'linear_sum': linear_sum,
                    'square_sum': square_sum,
                    'num_nodes': num_nodes,
                    'radius': torch.sqrt(((cluster_points - center) ** 2).sum(dim=1).max())  # 初始半径备用
                })

                micro_clusters[cluster_id] = {
                    'center': center,
                    'linear_sum': linear_sum,
                    'square_sum': square_sum,
                    'num_nodes': num_nodes,
                    'radius': radius
                }

            if int(label) not in self.mc:
                self.mc[int(label)] = {}

            # 将 micro_clusters 的内容添加到 self.mc[int(label)] 的末尾
            max_cluster_id = max(self.mc[int(label)].keys(), default=-1)

            # 更新 micro_clusters 的 cluster_id，以便它们不会与现有的 cluster_id 冲突
            updated_micro_clusters = {cluster_id + max_cluster_id + 1: info for cluster_id, info in micro_clusters.items()}

            # 将更新后的 micro_clusters 添加到 self.mc[int(label)]
            self.mc[int(label)].update(updated_micro_clusters)

        return self.mc

    def update_prototypes(self):
        """
        根据微簇信息更新每个类别的原型。

        输出:
        - self.prototypes (dict): 一个包含每个类别原型信息的字典，其中键是类别标签，值是一个字典，包含:
            - 'class_prototype' (tensor): 类别的总原型向量
            - 'sub_prototypes' (list[tensor]): 每个微簇的原型向量列表
            - 'densities' (list[float]): 每个微簇的样本密度列表
        """
        self.prototypes = {}

        for lbl, clusters in self.mc.items():
            sub_prototypes = []
            densities = []
            total_nodes = sum(cluster['num_nodes'] for cluster in clusters.values())  # 计算该类别的总节点数

            if total_nodes == 0:
                raise ValueError(f"类别 {lbl} 中的总节点数为 0,无法更新原型。")

            # 遍历每个微簇，提取簇中心和密度信息
            for cluster_id, cluster in clusters.items():
                sub_prototypes.append(cluster['center'])  # 簇中心
                densities.append(cluster['num_nodes'] / total_nodes)  # 样本密度为该簇节点数占总节点数的比例

            # 按密度加权求和计算类别的总原型
            class_prototype = sum(p * d for p, d in zip(sub_prototypes, densities))

            # 更新类别的原型信息
            self.prototypes[lbl] = {
                'class_prototype': class_prototype,
                'sub_prototypes': sub_prototypes,
                'densities': densities,
            }

        return self.prototypes


    def update_radius_on_stats(self, cluster):
        """
        根据微簇的线性和、平方和以及样本数量动态计算和更新微簇的半径
        """
        num_points = cluster['num_nodes']

        # 防止除以0的情况
        if num_points <= 1:
            return cluster['radius']

        linear_sum = cluster['linear_sum']
        square_sum = cluster['square_sum']

        # 计算方差
        var = (square_sum / num_points) - (linear_sum / num_points) ** 2

        # 方差应为正数，防止浮点数误差导致负值
        var = torch.clamp(var, min=1e-5)

        # 根据方差的平方根计算新的半径，α为缩放因子
        alpha = 1.0
        new_radius = alpha * torch.sqrt(var).mean()  # 取均值处理多维情况

        return new_radius

    def update_mc_layer(self, embeds, lbs):
        """
        更新微簇层。
        - 如果存在新类别，初始化其微簇。
        - 对已有类别的嵌入进行微簇更新。

        输入:
        - embeds: 嵌入向量 (Tensor)
        - lbs: 对应的标签 (Tensor)

        输出:
        - self.mc: 更新后的微簇信息
        """
        unique_lbs = lbs.unique()
        # 检查是否存在新的类别
        new_labels = [label for label in unique_lbs if label.item() not in self.mc]
        # (f"new labels detected:{new_labels}")
        if new_labels:
            # 初始化新类别的微簇
            mask = torch.isin(lbs, torch.tensor(new_labels))
            initial_embeds = embeds[mask]
            initial_lbs = lbs[mask]
            self.initialize_microcluster(initial_embeds, initial_lbs)

            # 剩余部分更新已有类别
            remaining_embeds = embeds[~mask]
            remaining_lbs = lbs[~mask]
            self.update_microcluster(remaining_embeds, remaining_lbs)
        else:
            # 仅更新已有类别的微簇
            self.update_microcluster(embeds, lbs)

        return self.mc

    def update_microcluster(self, embeds, lbs):
        """
        更新微簇信息。
        - 将嵌入按标签分组。
        - 对每个标签的嵌入进行微簇更新。
        - 无法归入现有微簇的嵌入将初始化为新的微簇。

        输入:
        - embeds: 嵌入向量 (Tensor)
        - lbs: 对应的标签 (Tensor)

        输出:
        - self.mc: 更新后的微簇信息
        """
        all_remaining_embeds = []
        all_remaining_lbs = []

        for label in lbs.unique():
            # 提取当前标签的嵌入
            label_mask = lbs == label
            remaining_embeds = embeds[label_mask]

            # 获取当前标签的微簇
            clusters = self.mc[int(label)]
            
            # 更新微簇，返回未分配的嵌入
            remaining_embeds = self.update_clusters(clusters, remaining_embeds)

            # 保存未分配嵌入和其对应标签
            all_remaining_embeds.append(remaining_embeds)
            all_remaining_lbs.extend([label.item()] * len(remaining_embeds))

        # 将未分配的嵌入初始化为新的微簇
        if all_remaining_embeds:
            all_remaining_embeds = torch.cat(all_remaining_embeds, dim=0)
            all_remaining_lbs = torch.tensor(all_remaining_lbs, dtype=torch.int64)
            self.initialize_microcluster(all_remaining_embeds, all_remaining_lbs)

        return self.mc

    def update_clusters(self, clusters, remaining_embeds):
        """
        将嵌入分配到现有微簇。
        - 如果嵌入到某微簇的距离小于微簇半径，则归入该微簇。

        输入:
        - clusters: 当前类别的微簇信息（字典 -> 字典结构）
        - remaining_embeds: 待分配的嵌入向量

        输出:
        - 未分配的嵌入向量
        """
        # 提取所有微簇的中心和半径
        cluster_centers = torch.cat([cluster['center'].unsqueeze(0) for cluster in clusters.values()], dim=0).to('cuda:0')
        cluster_radii = torch.tensor([cluster['radius'] for cluster in clusters.values()]).to('cuda:0')
        
        # 计算嵌入与每个微簇中心的距离
        distances = torch.cdist(remaining_embeds, cluster_centers)
        min_distances, min_indices = distances.min(dim=1)

        # 找到距离在半径内的嵌入
        to_update_mask = min_distances <= cluster_radii[min_indices]

        # 更新每个符合条件的微簇
        for embed, idx in zip(remaining_embeds[to_update_mask], min_indices[to_update_mask]):
            cluster_key = list(clusters.keys())[idx.item()]  # 根据索引获取微簇键
            self.update_cluster_stats(embed, clusters[cluster_key])

        # 返回未分配的嵌入
        return remaining_embeds[~to_update_mask]


    def update_cluster_stats(self, embed, cluster):
        """
        更新单个微簇的统计信息。
        - 更新线性和、平方和、样本数、中心和半径。

        输入:
        - embed: 新的嵌入向量
        - cluster: 待更新的微簇信息
        """
        cluster['linear_sum'] += embed
        cluster['square_sum'] += embed ** 2
        cluster['num_nodes'] += 1
        cluster['center'] = cluster['linear_sum'] / cluster['num_nodes']
        cluster['radius'] = self.update_radius_on_stats(cluster)


"""def test_memory():
    # 初始化 Memory 实例
    memory = Memory(minPts=2)

    # 生成模拟数据
    novel_embeds = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35],
        [0.9, 0.8, 0.7],
        [0.92, 0.82, 0.72],
        [0.5, 0.5, 0.5]
    ], dtype=torch.float32)

    novel_lbs = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int64)

    # 初始化微簇
    microclusters = memory.update_mc_layer(novel_embeds, novel_lbs)
    print("Microclusters after initialization:")
    for label, clusters in microclusters.items():
        print(f"Label {label}:")
        for cluster_id, cluster_info in clusters.items():
            print(f"  Cluster {cluster_id}: {cluster_info}")

    # 更新原型
    prototypes = memory.update_prototypes()
    print("\nPrototypes after updating:")
    for label, prototype_info in prototypes.items():
        print(f"Label {label}: {prototype_info}")

if __name__ == "__main__":
    test_memory()"""


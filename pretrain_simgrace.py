from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.optim as optim
from copy import deepcopy
from torch.optim import Adam
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.loader import DataLoader
from torch.autograd import Variable

from utils import *
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_name', default='Amazon_clothing',
                        help='Dataset:Amazon_clothing/cora-full/Computers/coauthorCS')
    parser.add_argument('--pretrain_model', required=False, help='Existing model path.')
    parser.add_argument('--overwrite_pretrain', action='store_true', help='Delete existing pre-train model')
    parser.add_argument('--output_path', default='./pretrain_model_simgrace', help='Path for output pre-trained model.')
    parser.add_argument('--batch_size', type=int, default=256, help='Input batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for dataset loading')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 3).')
    parser.add_argument('--num_layers', type=int, default=1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--gnn_type', type=str, default="GAT",
                        help='We support gnn like \GCN\ \GAT\ \GT\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    args = parser.parse_args()
    return args


class SimGRACE(torch.nn.Module):

    def __init__(self, gnn_type='TransformerConv', dataset_name='Cora', hid_dim=64, gln=2, num_epoch=100,
                 device: int = 0):  # hid_dim=16
        super().__init__()
        self.args = get_args()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim = hid_dim
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)

    def initialize_gnn(self, input_dim, hid_dim):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            # self.gnn = Encoder(in_channels=input_dim,hidden_channels=hid_dim,encoder_type='GCN')
            self.gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

    def load_graph_data(self):

        data, id_by_class, base_id, novel_id, node_list, num_all_nodes = load_raw_data(
            self.dataset_name)

        novel_train_id, novel_test_id = split_novel_data(novel_id, self.dataset_name)

        pretrain_id = base_id + novel_train_id
        cache = {"pretrain_seed": args.seed, "id_by_class": id_by_class, "base_id": base_id,
                 "novel_id": novel_id, "novel_train_id": novel_train_id, "novel_test_id": novel_test_id}

        cache_path = os.path.join("./cache", (str(self.dataset_name) + "_" + str(self.gnn_type) + ".pkl"))
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        save_object(cache, cache_path)
        del cache

        node_labels = data.y
        pretrain_classes = torch.tensor(pretrain_id, dtype=torch.long)
        print(f"pretrain ids are:{pretrain_classes}")

        pretrain_node_mask = torch.isin(node_labels, pretrain_classes)  # bool mask
        pretrain_node_indices = pretrain_node_mask.nonzero(as_tuple=True)[0]

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)

        edge_index, _ = subgraph(pretrain_node_indices, edge_index, relabel_nodes=True, num_nodes=num_all_nodes)
        x = x[pretrain_node_indices]
        data = Data(x=x, edge_index=edge_index)

        self.graph_list = list(ClusterData(data=data, num_parts=300))
        self.input_dim = data.x.shape[1]

    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")
        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers)
        return loader

    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean()
        return loss

    def perturbate_gnn(self, data):
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2

    def train_simgrace(self, loader, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data)
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            # print(f"Step {step}, Loss: {loss.item()}")  # 打印每一步的损失

            loss.backward()
            optimizer.step()

            """# 打印梯度信息
            for name, param in self.named_parameters():
                if param.grad is not None:
                    print(f"{name} - Gradient norm: {param.grad.norm().item()}")"""

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.01, decay=0.0001, epochs=100):

        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_simgrace(loader, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./pre_trained_gnn/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./pre_trained_gnn/{}/{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE',
                                                                      self.gnn_type,
                                                                      str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed)
    mkdir('./pre_trained_gnn/')

    pt = SimGRACE(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hidden, gln=args.num_layer,
                  num_epoch=args.epochs, device=args.device)

    pt.pretrain(batch_size=args.batch_size, lr=args.lr, decay=args.weight_decay, epochs=args.epochs)
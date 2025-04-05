import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import global_mean_pool, GATConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from utils import act


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set

        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """
        GraphConv = GATConv

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        if batch == None:
            return node_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool='mean', encoder_type='GAT'):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        if encoder_type == 'GCN':
            self.conv1 = GCNConv(in_channels, self.hidden_channels)
        elif encoder_type == 'GAT':
            self.conv1 = GATConv(in_channels, self.hidden_channels)
        elif encoder_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, self.hidden_channels)
        elif encoder_type == 'SGC':
            self.conv1 = SGConv(in_channels, self.hidden_channels)
        elif encoder_type == 'GIN':
            self.mlp1 = nn.Linear(in_channels, self.hidden_channels)
            self.conv1 = GINConv(self.mlp1)

        self.prelu1 = nn.PReLU(self.hidden_channels)
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, edge_index, batch=None):

        x1 = self.conv1(x, edge_index)
        x1 = self.prelu1(x1)
        x1 = F.normalize(x1)
        if batch == None:
            return x1
        else:
            x1 = self.pool(x1, batch.long())
            return x1


class Projector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x)
        return x

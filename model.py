import numpy as np
import os
import networkx as nx
from operator import itemgetter
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from numpy.random import randn
import random
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch_scatter import scatter_sum

from torch_geometric.utils import from_networkx, degree, sort_edge_index, to_networkx
from torch_geometric.nn import GATConv, GraphConv, GCNConv, GINConv, GINEConv, Set2Set, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool, LayerNorm, BatchNorm, GlobalAttention



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.activation = activation
#         self.bn = torch.nn.BatchNorm1d(hidden_dim)
        

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
#         x = self.bn(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

class SizeNorm(torch.nn.Module):
    def __init__(self):
        super(SizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1) 

class GNNvirtual(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GNNvirtual, self).__init__()
        
        self.num_visible = num_nodes
        
        self.depth = depth
        self.dropout = dropout
        
#         self.lin = MLP(1, hidden, hidden, dropout, F.relu)
        
        self.conv = torch.nn.ModuleList()
        for i in range(self.depth):
            self.conv.append(GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)), eps=0.1, train_eps=True))
            
        self.lns = torch.nn.ModuleList()
        self.sizenorm = torch.nn.ModuleList()
        for i in range(self.depth):
            self.lns.append(LayerNorm(hidden))
            self.sizenorm.append(SizeNorm())
            
        self.mlp = MLP(hidden, hidden, 1, dropout, F.relu)
        
        self.node_mlp = MLP(1, hidden, hidden, dropout, F.relu)
        self.edge_mlp = MLP(2, hidden, hidden, dropout, F.relu)
#         self.appnp = APPNP(3,0.1)
        
        self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(hidden, 2*hidden), torch.nn.BatchNorm1d(2*hidden), torch.nn.ReLU(), torch.nn.Linear(2*hidden, 1)))
        
       
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        
        
        for layer in range(self.depth):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden), torch.nn.ReLU(), \
                                                    torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden), torch.nn.ReLU()))

    def psi(self, config):
        
        data.x = config.to(torch.float32).view(-1,1)
        
#         data = data.cuda()
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # obtain node embedding
#         
        
#         x = torch.ones(data.num_nodes, 1).to(device)
        
#         print(edge_index)
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
        edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0) #.to(device)
#         edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        x = self.node_mlp(x)
        edge_attr = self.edge_mlp(edge_attr)
        
        z = []

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
                                                           
        for i in range(self.depth):
            h = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x, edge_index, edge_attr)
#             z.append(global_add_pool(x, batch))
            x = self.sizenorm[i](x, batch)
            x = self.lns[i](x)
        
#             if i < self.depth - 1:
#                 x = F.relu(x)
            x = F.relu(x)
        
            x = x + h
#             z.append(x)
            
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.dropout, training = self.training)
            x = x + virtualnode_embedding[batch]
            
            z.append(global_add_pool(x, batch))

        # readout
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat((x_mean, x_max), 1)

#         x = 0
#         for i in range(self.depth):       
#             x = x + z[i]
#         x = global_add_pool(x, batch)
        x = self.pool(x, batch)
#         z[-1] = x
#         x = torch.sum(torch.cat(z, 0), 0)    
        
        # regression
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp(x)
        x = torch.sigmoid(x)
#         if self.training:
#             return output
#         else:
#             # At inference time, relu is applied to output to ensure positivity
#             return torch.clamp(output, min=float('-inf'), max=0)
        return x

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class CNN2D(nn.Module):
    def __init__(self, num_visible, num_hidden, filter_num=4, kernel_size=3):
        super(CNN2D, self).__init__()
        
        self.num_visible = num_visible
        self.length = 10
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.inf = 1e20
        self.log_inf = 40

        self.net = nn.Sequential(
            nn.Conv2d(1, filter_num, kernel_size=kernel_size),
            nn.MaxPool2d(2, 2),
            nn.ConvTranspose2d(filter_num, 1, kernel_size=2, stride=2),
        #    nn.Tanh(),
        )

    def handmade_padding(self, spin_lattice):
        """
        Args:
            spin_lattice: [length, length]
        Returns:
            spin_lattice_padded: [length + kernel_size - 1, length + kernel_size - 1]
        """
        spin_lattice_padded = torch.zeros(self.length + self.kernel_size - 1, self.length + self.kernel_size - 1).cuda()
        half_k = int((self.kernel_size - 1) / 2)
        spin_lattice_padded[half_k:half_k + self.length, half_k:half_k + self.length] = spin_lattice[:, :]
        spin_lattice_padded[half_k:half_k + self.length, :half_k] = spin_lattice[:, -half_k:]
        spin_lattice_padded[half_k:half_k + self.length, -half_k:] = spin_lattice[:, :half_k]
        spin_lattice_padded[:half_k, :] = spin_lattice_padded[self.length:half_k + self.length, :]
        spin_lattice_padded[-half_k:, :] = spin_lattice_padded[half_k:half_k * 2, :]
        return spin_lattice_padded

    def psi(self, data, v):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        # assert v.size()[0] == self.length ** 2
        # v = v.view(self.length, self.length).float()
        # v = self.handmade_padding(v)
        # v = self.net(v.view(1, 1, self.length + self.kernel_size - 1, self.length + self.kernel_size - 1))
        sign_v, v = self.log_psi(data, v)
        v = torch.minimum(v, torch.full_like(v, fill_value=self.log_inf))
        return v.exp() * sign_v

    def log_psi(self, data, v):
        v = v.view(self.length, self.length).float()
        v = self.handmade_padding(v)
        v = self.net(v.view(1, 1, self.length + self.kernel_size - 1, self.length + self.kernel_size - 1))
        sign_v = v.sign()
        v = v * sign_v
        return sign_v.prod(), v.log().sum()

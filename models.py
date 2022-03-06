import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.random import randn
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, ModuleDict
from torch_scatter import scatter_sum, scatter_mean

from torch_geometric.utils import from_networkx, degree, sort_edge_index
from torch_geometric.nn import GATConv, GraphConv, GCNConv, GINConv, GINEConv, Set2Set, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool, LayerNorm, BatchNorm, GlobalAttention
from torch_geometric.data import Data, Batch

class HexagonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(HexagonConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.ones_like(self.conv.weight)
        if mask.shape[-1] == 3:
            row = torch.tensor([[0], [2]])
            col = torch.tensor([[2], [0]])
        elif mask.shape[-1] == 5:
            row = torch.tensor([[0], [0], [1], [-2], [-1], [-1]])
            col = torch.tensor([[-2], [-1], [-1], [0], [0], [1]])
        mask[:,:,row,col] = 0
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class Hexagon108Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(Hexagon108Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=0, padding_mode=padding_mode, bias=bias)

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:,:,0,1:8] = x[:,:,12,7:14]
        for i in range(3):
            x[:,:,i,8+i] = x[:,:,6+i,2+i]
        for i in range(4):
            x[:,:,3+i,10+i] = x[:,:,9+i,4+i]
        for i in range(3):
            x[:,:,7+i,13] = x[:,:,1+i,1]
        for i in range(3):
            x[:,:,10+i,14] = x[:,:,4+i,2]
        for i in range(8):
            x[:,:,13,7+i] = x[:,:,1,1+i]
        for i in range(4):
            x[:,:,9+i,3+i] = x[:,:,3+i,9+i]
        for i in range(3):
            x[:,:,6+i,1+i] = x[:,:,0+i,7+i]
        for i in range(2):
            x[:,:,4+i,1] = x[:,:,10+i,13]
        for i in range(4):
            x[:,:,0+i,0] = x[:,:,6+i,12]

        # conv
        x = self.conv(x)

        # postprocessing
        for i in range(5):
            for j in range(8+i, 13):
                x[:,:,i,j] = 0
        for i in range(4):
            x[:,:,2+i,9+i] = 0
        x[:, :, 6:9, 12] = 0

        for i in range(6):
            for j in range(i+1):
                x[:,:,6+i,j] = 0
        for i in range(3):
            x[:,:,5+i,0+i] = 0
        x[:, :, 3:5, 0] = 0

        return x

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super(MaskedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.ones_like(self.conv.weight)
        if mask.shape[-1] == 3:
            row = torch.tensor([[0], [0], [2], [2]])
            col = torch.tensor([[0], [2], [0], [2]])
        elif mask.shape[-1] == 5:
            row = torch.tensor([[0], [0], [0], [0], [1], [1], [3], [3], [4], [4], [4], [4]])
            col = torch.tensor([[0], [1], [3], [4], [0], [4], [0], [4], [0], [1], [3], [4]])
        mask[:,:,row,col] = 0
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(UpConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[0], [0], [1], [2], [2]])
        col = torch.tensor([[0], [1], [1], [1], [2]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class LeftConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(LeftConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[1], [2], [2], [2], [3]])
        col = torch.tensor([[1], [0], [1], [2], [1]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

class RightConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(RightConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                  stride=2, padding=padding, padding_mode=padding_mode, bias=bias)

        mask = torch.zeros_like(self.conv.weight)

        row = torch.tensor([[1], [2], [2], [2], [3]])
        col = torch.tensor([[1], [1], [2], [3], [3]])
        mask[:,:,row,col] = 1
        self.register_buffer('mask', mask)

    def _mask_conv(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)

    def forward(self, x):
        self._mask_conv()
        x = self.conv(x)
        return x

## temporary version
class CNN2D(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
        for i in range(2):
            self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1))

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
        self.In_list = nn.ModuleList()
        for i in range(2):
            self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))

        self.log_psi = 0
        self.arg_psi = 0

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        first_out = config
        for i in range(len(self.conv_list)):
            config = self.conv_list[i](config)
            # if (i+1) != len(self.conv_list):
            config = self.In_list[i](config)
            config = F.relu(config)
        # config = self.conv_list[0](config)
        # config = F.relu(config)
        # config = self.conv_list[1](config)

        config = config + first_out
        # config = F.relu(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)
        # config = self.bn3(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_explore(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_explore, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
        for i in range(2):
            self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1))

        # self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        # self.linear2 = nn.Linear(self.num_visible, 2)

        self.linear1 = nn.Linear(self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.filters, self.num_visible)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
        self.In_list = nn.ModuleList()
        for i in range(2):
            self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))

        # SE layers
        self.fc1 = nn.Conv2d(self.filters, self.filters//2, kernel_size=1)
        self.fc2 = nn.Conv2d(self.filters//2, self.filters, kernel_size=1)

        self.log_psi = 0
        self.arg_psi = 0

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        first_out = config
        for i in range(len(self.conv_list)):
            config = self.conv_list[i](config)
            # if (i+1) != len(self.conv_list):
            config = self.In_list[i](config)
            config = F.relu(config)
        # config = self.conv_list[0](config)
        # config = F.relu(config)
        # config = self.conv_list[1](config)

        # Squeeze
        w = F.avg_pool2d(config, config.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        config = config * w

        config = config + first_out
        # config = F.relu(config)

        # config = config.view(config.size(0), -1)
        config = F.avg_pool2d(config, config.size(2))
        # config = self.linear1(config)
        # # config = self.bn3(config)
        # config = F.relu(config)
        # # config = F.sigmoid(config)
        # out = self.linear2(config)  # output log(|psi|) and arg(psi)
        config = config.squeeze()
        config_log = self.linear1(config)
        config_arg = self.linear2(config)

        out1 = torch.sum(config_log, dim=1, keepdim=True)
        out2 = torch.sum(config_arg, dim=1, keepdim=True)
        self.log_psi = out1 + 0 * out2
        self.arg_psi = out2 + 0 * out1
        # print(self.log_psi.shape)
        # print(self.arg_psi.shape)
        # self.log_psi = out[:, 0:1]
        # self.arg_psi = out[:, 1:]
        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

# class CNN2D_SE(torch.nn.Module):
#     def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
#         super(CNN2D_SE, self).__init__()
#
#         self.num_visible = num_visible
#         self.kernel_size = kernel_size
#         self.filters_in = filters_in
#         self.filters = filters
#         self.height = int(np.sqrt(self.num_visible))
#
#         self.conv_list = nn.ModuleList()
#         self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')
#         for i in range(2):
#             self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
#
#         self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
#         self.linear2 = nn.Linear(self.num_visible, 2)
#
#         self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
#         self.In_list = nn.ModuleList()
#         for i in range(2):
#             self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))
#
#         # SE layers
#         self.fc1 = nn.Conv2d(self.filters, self.filters//2, kernel_size=1)
#         self.fc2 = nn.Conv2d(self.filters//2, self.filters, kernel_size=1)
#
#         self.Ln = nn.LayerNorm([self.num_visible])
#
#         self.log_psi = 0
#         self.arg_psi = 0
#
#     def psi(self, data, config_in):
#         # batch, x, edge_index = data.batch, data.x, data.edge_index
#         config = config_in.clone().float()
#         config = config.view(-1, 1, self.height, self.height)
#
#         config = self.first_conv(config)
#         config = self.first_In(config)
#         config = F.relu(config)
#
#         first_out = config
#         for i in range(len(self.conv_list)):
#             config = self.conv_list[i](config)
#             # if (i+1) != len(self.conv_list):
#             config = self.In_list[i](config)
#             config = F.relu(config)
#         # config = self.conv_list[0](config)
#         # config = F.relu(config)
#         # config = self.conv_list[1](config)
#
#         # Squeeze
#         w = F.avg_pool2d(config, config.size(2))
#         w = F.relu(self.fc1(w))
#         w = torch.sigmoid(self.fc2(w))
#         # Excitation
#         config = config * w
#
#         config = config + first_out
#         # config = F.relu(config)
#
#         config = config.view(config.size(0), -1)
#         config = self.linear1(config)
#         # config = self.bn3(config)
#
#         config = self.Ln(config)
#         config = F.relu(config)
#         # config = F.sigmoid(config)
#         out = self.linear2(config)  # output log(|psi|) and arg(psi)
#
#         self.log_psi = out[:, 0:1]
#         self.arg_psi = out[:, 1:]
#         psi_value = (self.log_psi + 1j * self.arg_psi).exp()
#         return psi_value
#
#     def psi_batch(self, data, config):
#         return self.psi(data, config)

class CNN2D_SE(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_SE, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock, nn.Conv2d, filters, num_blocks=2, height=self.height, stride=1)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])

        self.Ln = nn.LayerNorm([self.num_visible])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)

def first_pad_config(config):
    config = config.clone()
    # print(config.shape)
    pad = torch.nn.ZeroPad2d((1,2,1,2))
    config = pad(config)
    # print(config.shape)
    config[:,:,5,7] = config[:,:,5,1]
    config[:,:,6,7] = config[:,:,6,1]
    config[:,:,5,1] = 0
    config[:,:,6,1] = 0
    config[:,:,7,5] = config[:,:,1,5]
    config[:,:,7,6] = config[:,:,1,6]
    config[:,:,1,5] = 0
    config[:,:,1,6] = 0

    config[:,:,0,0] = config[:,:,0+6,0+6]
    config[:,:,0,1] = config[:,:,0+6,1+6]
    config[:,:,0,2] = config[:,:,0+6,2]
    config[:,:,0,3] = config[:,:,0+6,3]
    config[:,:,0,4] = config[:,:,0+6,4]
    config[:,:,1,5] = config[:,:,1+6,5]
    config[:,:,1,6] = config[:,:,1+6,6]
    config[:,:,2,7] = config[:,:,2,7-6]
    config[:,:,3,7] = config[:,:,3,7-6]
    config[:,:,4,7] = config[:,:,4,7-6]
    config[:,:,5,8] = config[:,:,5,8-6]
    config[:,:,6,8] = config[:,:,6,8-6]
    config[:,:,7,8] = config[:,:,7-6,8-6]
    config[:,:,7,7] = config[:,:,7-6,7-6]
    config[:,:,8,7] = config[:,:,8-6,7-6]
    config[:,:,8,6] = config[:,:,8-6,6]
    config[:,:,8,5] = config[:,:,8-6,5]
    config[:,:,7,4] = config[:,:,7-6,4]
    config[:,:,7,3] = config[:,:,7-6,3]
    config[:,:,7,2] = config[:,:,7-6,2]
    config[:,:,6,1] = config[:,:,6,1+6]
    config[:,:,5,1] = config[:,:,5,1+6]
    config[:,:,4,0] = config[:,:,4,0+6]
    config[:,:,3,0] = config[:,:,3,0+6]
    config[:,:,2,0] = config[:,:,2,0+6]
    config[:,:,1,0] = config[:,:,1+6,0+6]

    return config

def pad_config(config):
    config = config.clone()
    pad = torch.nn.ZeroPad2d((1,1,1,1))
    config = pad(config)

    config[:,:,1,7] = 0

    config[:,:,0,0] = config[:,:,0+6,0+6]
    config[:,:,0,1] = config[:,:,0+6,1+6]
    config[:,:,0,2] = config[:,:,0+6,2]
    config[:,:,0,3] = config[:,:,0+6,3]
    config[:,:,0,4] = config[:,:,0+6,4]
    config[:,:,1,5] = config[:,:,1+6,5]
    config[:,:,1,6] = config[:,:,1+6,6]
    config[:,:,2,7] = config[:,:,2,7-6]
    config[:,:,3,7] = config[:,:,3,7-6]
    config[:,:,4,7] = config[:,:,4,7-6]
    config[:,:,5,8] = config[:,:,5,8-6]
    config[:,:,6,8] = config[:,:,6,8-6]
    config[:,:,7,8] = config[:,:,7-6,8-6]
    config[:,:,7,7] = config[:,:,7-6,7-6]
    config[:,:,8,7] = config[:,:,8-6,7-6]
    config[:,:,8,6] = config[:,:,8-6,6]
    config[:,:,8,5] = config[:,:,8-6,5]
    config[:,:,7,4] = config[:,:,7-6,4]
    config[:,:,7,3] = config[:,:,7-6,3]
    config[:,:,7,2] = config[:,:,7-6,2]
    config[:,:,6,1] = config[:,:,6,1+6]
    config[:,:,5,1] = config[:,:,5,1+6]
    config[:,:,4,0] = config[:,:,4,0+6]
    config[:,:,3,0] = config[:,:,3,0+6]
    config[:,:,2,0] = config[:,:,2,0+6]
    config[:,:,1,0] = config[:,:,1+6,0+6]

    return config

def reorganize(config):
    config = config.clone()
    config[:,:,5-1,1-1] = config[:,:,5-1,7-1]
    config[:,:,6-1,1-1] = config[:,:,6-1,7-1]
    config[:,:,1-1,5-1] = config[:,:,7-1,5-1]
    config[:,:,1-1,6-1] = config[:,:,7-1,6-1]

    config = config[:,:,0:6,0:6]

    return config

class CNN2D_SE_Hex_new(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_SE_Hex_new, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=0, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock_new, HexagonConv2d, filters, num_blocks=2, height=self.height+1, stride=1)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.height * self.height * self.filters, self.height * self.height)
        self.linear2 = nn.Linear(self.height * self.height, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height+1, self.height+1])

        self.Ln = nn.LayerNorm([self.height * self.height])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = first_pad_config(config)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)


        config = reorganize(config)
        # print(config.shape)
        config = config.reshape(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)


class SEResidualBlock_new(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1):
        super(SEResidualBlock_new, self).__init__()
        self.conv1 = nn.Sequential(
            conv(filters_in, filters, kernel_size=3, stride=stride, padding=0, bias=True, padding_mode='circular'),
            nn.LayerNorm([filters, height, height]),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            conv(filters, filters, kernel_size=3, stride=1, padding=0, bias=True, padding_mode='circular'),
            nn.LayerNorm([filters, height, height]),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

    def forward(self, x):
        out = pad_config(x)
        out = self.conv1(out)
        out = pad_config(out)
        out = self.conv2(out)
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)

        return out

class CNN2D_SE_Hex_108(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, non_local=False, mode='embedded'):
        super(CNN2D_SE_Hex_108, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.shape = (12, 13)

        self.conv_list = nn.ModuleList()
        self.first_conv = Hexagon108Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock, Hexagon108Conv2d, filters, num_blocks=2, height=self.shape, stride=1, non_local=non_local, mode=mode)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()

        config = torch.zeros((config_in.shape[0], self.shape[0] * self.shape[1])).cuda()
        cnt = 0
        for i in range(8):
            config[:, self.shape[1] * 0 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 3 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(10):
            config[:, self.shape[1] * 4 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(11):
            config[:, self.shape[1] * 5 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(10):
            config[:, self.shape[1] * 6 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 7 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 8 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(9):
            config[:, self.shape[1] * 9 + 4 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 10 + 5 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(7):
            config[:, self.shape[1] * 11 + 6 + i] = config_in[:, cnt]
            cnt += 1


        config = config.view(-1, 1, self.shape[0], self.shape[1])

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi
        # return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_SE_Hex(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, first_kernel_size=3, non_local=False, mode='embedded'):
        super(CNN2D_SE_Hex, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))
        self.shape = (self.height, self.height)

        self.conv_list = nn.ModuleList()
        if first_kernel_size == 3:
            self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        elif first_kernel_size == 5:
            self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=5, stride=1, padding=2,
                                            padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.shape, stride=1, non_local=non_local, mode=mode)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        # if non_local:
        #     layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode='embedded'))
            # layers.append(block(conv, self.filters, filters, height, stride))

        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi
        # return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_SE_Hex_FCN(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_SE_Hex_FCN, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = HexagonConv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.filters, self.filters)
        self.linear2 = nn.Linear(self.filters, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])

        self.Ln = nn.LayerNorm([self.filters])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = F.avg_pool2d(config, config.shape[2], config.shape[3]).squeeze()
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)

class SEResidualBlock(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
            nn.LayerNorm([filters, height[0], height[1]]),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.LayerNorm([filters, height[0], height[1]]),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)

        return out

class NonLocalBlock(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, mode='embedded'):
        super(NonLocalBlock, self).__init__()

        self.mode = mode
        self.filters = filters

        self.conv1 = nn.Sequential(
            conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
            # nn.LayerNorm([filters, height[0], height[1]]),
            # nn.ReLU(inplace=True),
            )

        self.W_z = nn.Sequential(
            conv(filters, filters_in, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.LayerNorm([filters_in, height[0], height[1]]),
        )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        if self.mode == 'embedded' or self.mode == "concatenate":
            self.phi = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)
            self.theta = conv(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.filters * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )

        # self.shortcut = nn.Sequential()
        # if stride != 1 or filters_in != filters:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
        #         nn.LayerNorm([filters, height, height])
        #     )


    def forward(self, x):

        batch_size = x.size(0)
        # N C HW
        g = self.conv1(x).view(batch_size, self.filters, -1)
        g = g.permute(0, 2, 1)

        if self.mode == 'embedded':
            theta_x = self.theta(x).view(batch_size, self.filters, -1)
            phi_x = self.phi(x).view(batch_size, self.filters, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.filters, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.filters, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)

        elif self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N

        out = torch.matmul(f_div_C, g)

        # contiguous here just allocates contiguous chunk of memory
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.filters, *x.size()[2:])

        out = self.W_z(out)
        # residual connection
        out = out + x
        # out = F.relu(out)

        return out

class Kagome12Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome12Conv2D, self).__init__()

        self.up_conv = UpConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular')
        self.left_conv = LeftConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular')
        self.right_conv = RightConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='circular')

    def forward(self, x):

        # preprocessing
        # pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        # x = pad(x)
        # x[:, :, 0, 0] = x[:, :, 8, 4]
        # x[:, :, 0, 1] = x[:, :, 8, 5]
        # x[:, :, 0, 2] = x[:, :, 8, 6]
        # x[:, :, 0, 3] = x[:, :, 8, 7]
        # x[:, :, 1, 5] = x[:, :, 5, 1]
        # x[:, :, 2, 6] = x[:, :, 6, 2]
        # x[:, :, 3, 7] = x[:, :, 7, 3]
        # x[:, :, 4, 8] = x[:, :, 8, 4]
        # x[:, :, 6, 9] = x[:, :, 2, 1]
        # x[:, :, 7, 9] = x[:, :, 3, 1]
        # x[:, :, 8, 9] = x[:, :, 4, 1]
        # x[:, :, 9, 9] = x[:, :, 5, 1]
        # x[:, :, 9, 7] = x[:, :, 1, 3]
        # x[:, :, 9, 5] = x[:, :, 1, 1]
        # x[:, :, 8, 3] = x[:, :, 4, 7]
        # x[:, :, 6, 1] = x[:, :, 2, 5]
        # x[:, :, 4, 0] = x[:, :, 8, 8]
        # x[:, :, 2, 0] = x[:, :, 6, 8]



        up = x.clone()
        left = x.clone()
        right = x.clone()
        up = self.up_conv(up)
        left = self.left_conv(left)
        right = self.right_conv(right)

        zeros = torch.zeros_like(up)

        outputs_up = torch.stack([up.T, zeros.T], dim=1)
        outputs_up = torch.flatten(outputs_up, start_dim=0, end_dim=1).T

        outputs_lr = torch.stack([left.T, right.T], dim=1)
        outputs_lr = torch.flatten(outputs_lr, start_dim=0, end_dim=1).T

        outputs = torch.stack([outputs_up, outputs_lr], dim=3)
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)


        # postprocessing
        # row = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [5], [6], [7], [7], [7]])
        # col = torch.tensor([[4], [6], [5], [6], [7], [6], [7], [0], [0], [0], [1], [2]])
        # outputs[:,:,row,col] = 0

        return outputs

class KagomeConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(KagomeConv2D, self).__init__()

        self.up_conv = UpConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.left_conv = LeftConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.right_conv = RightConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 0, 0] = x[:, :, 8, 4]
        x[:, :, 0, 1] = x[:, :, 8, 5]
        x[:, :, 0, 2] = x[:, :, 8, 6]
        x[:, :, 0, 3] = x[:, :, 8, 7]
        x[:, :, 1, 5] = x[:, :, 5, 1]
        x[:, :, 2, 6] = x[:, :, 6, 2]
        x[:, :, 3, 7] = x[:, :, 7, 3]
        x[:, :, 4, 8] = x[:, :, 8, 4]
        x[:, :, 6, 9] = x[:, :, 2, 1]
        x[:, :, 7, 9] = x[:, :, 3, 1]
        x[:, :, 8, 9] = x[:, :, 4, 1]
        x[:, :, 9, 9] = x[:, :, 5, 1]
        x[:, :, 9, 7] = x[:, :, 1, 3]
        x[:, :, 9, 5] = x[:, :, 1, 1]
        x[:, :, 8, 3] = x[:, :, 4, 7]
        x[:, :, 6, 1] = x[:, :, 2, 5]
        x[:, :, 4, 0] = x[:, :, 8, 8]
        x[:, :, 2, 0] = x[:, :, 6, 8]



        up = x.clone()
        left = x.clone()
        right = x.clone()
        up = self.up_conv(up)
        left = self.left_conv(left)
        right = self.right_conv(right)

        zeros = torch.zeros_like(up)

        outputs_up = torch.stack([up.T, zeros.T], dim=1)
        outputs_up = torch.flatten(outputs_up, start_dim=0, end_dim=1).T

        outputs_lr = torch.stack([left.T, right.T], dim=1)
        outputs_lr = torch.flatten(outputs_lr, start_dim=0, end_dim=1).T

        outputs = torch.stack([outputs_up, outputs_lr], dim=3)
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)


        # postprocessing
        row = torch.tensor([[0], [0], [1], [1], [1], [2], [3], [5], [6], [7], [7], [7]])
        col = torch.tensor([[4], [6], [5], [6], [7], [6], [7], [0], [0], [0], [1], [2]])
        outputs[:,:,row,col] = 0

        return outputs

class Kagome108Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='zeros'):
        super(Kagome108Conv2D, self).__init__()

        self.up_conv = UpConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.left_conv = LeftConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')
        self.right_conv = RightConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0, bias=True, padding_mode='circular')

    def forward(self, x):

        # preprocessing
        pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        x = pad(x)
        x[:, :, 1, 3] = x[:, :, 13, 15]
        x[:, :, 1, 5] = x[:, :, 13, 5]
        x[:, :, 2, 7] = x[:, :, 14, 7]
        x[:, :, 3, 9] = x[:, :, 15, 9]
        x[:, :, 4, 10] = x[:, :, 16, 10]
        x[:, :, 4, 11] = x[:, :, 16, 11]
        x[:, :, 6, 13] = x[:, :, 6, 1]
        x[:, :, 7, 13] = x[:, :, 7, 1]
        x[:, :, 8, 14] = x[:, :, 8, 2]
        x[:, :, 10, 15] = x[:, :, 10, 3]
        x[:, :, 11, 15] = x[:, :, 11, 3]
        x[:, :, 12, 16] = x[:, :, 12, 4]
        x[:, :, 14, 15] = x[:, :, 2, 3]
        x[:, :, 14, 16] = x[:, :, 2, 4]
        x[:, :, 15, 15] = x[:, :, 3, 3]
        x[:, :, 16, 14] = x[:, :, 4, 2]
        x[:, :, 17, 13] = x[:, :, 5, 1]
        x[:, :, 17, 11] = x[:, :, 5, 11]
        x[:, :, 16, 9] = x[:, :, 4, 9]
        x[:, :, 15, 7] = x[:, :, 3, 7]
        x[:, :, 14, 6] = x[:, :, 2, 6]
        x[:, :, 14, 5] = x[:, :, 2, 5]
        x[:, :, 12, 3] = x[:, :, 12, 15]
        x[:, :, 10, 2] = x[:, :, 10, 14]
        x[:, :, 8, 1] = x[:, :, 8, 13]
        x[:, :, 6, 0] = x[:, :, 6, 12]
        x[:, :, 4, 0] = x[:, :, 16, 12]
        x[:, :, 4, 1] = x[:, :, 16, 13]
        x[:, :, 3, 1] = x[:, :, 15, 13]
        x[:, :, 2, 2] = x[:, :, 14, 14]


        # conv
        up = x.clone()
        left = x.clone()
        right = x.clone()
        up = self.up_conv(up)
        left = self.left_conv(left)
        right = self.right_conv(right)

        zeros = torch.zeros_like(up)

        outputs_up = torch.stack([up.T, zeros.T], dim=1)
        outputs_up = torch.flatten(outputs_up, start_dim=0, end_dim=1).T

        outputs_lr = torch.stack([left.T, right.T], dim=1)
        outputs_lr = torch.flatten(outputs_lr, start_dim=0, end_dim=1).T

        outputs = torch.stack([outputs_up, outputs_lr], dim=3)
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)


        # postprocessing
        for i in range(9):
            for j in range(7+i, 16):
                outputs[:,:,i,j] = 0


        for i in range(7):
            for j in range(i+1):
                outputs[:,:,9+i,j] = 0

        outputs[:, :, 0, 4:7] = 0
        outputs[:, :, 1, 6:8] = 0
        outputs[:, :, 2, 8] = 0
        outputs[:, :, 3, 9] = 0

        outputs[:, :, 6, 12] = 0
        outputs[:, :, 7, 13] = 0
        outputs[:, :, 8, 14] = 0

        outputs[:, :, 9, 14] = 0
        outputs[:, :, 10, 14] = 0
        outputs[:, :, 11, 15] = 0
        outputs[:, :, 13:, 14:] = 0
        outputs[:, :, 15, 13] = 0

        outputs[:, :, 15, 7:9] = 0
        outputs[:, :, 13, 5] = 0
        outputs[:, :, 14, 6] = 0
        outputs[:, :, 8, 0] = 0
        outputs[:, :, 9, 1] = 0
        outputs[:, :, 7, 0] = 0
        outputs[:, :, 3, 0] = 0
        outputs[:, :, 0:3, 0:2] = 0
        outputs[:, :, 0, 2] = 0

        return outputs

class CNN2D_SE_Kagome_108(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, non_local=False, mode='embedded'):
        super(CNN2D_SE_Kagome_108, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.shape = (16,16)

        self.conv_list = nn.ModuleList()
        self.first_conv = Kagome108Conv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlockKagome, Kagome108Conv2D, filters, num_blocks=2, height=self.shape, stride=1, non_local=non_local, mode=mode)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0] * self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config_in = config_in.clone().float()

        config = torch.zeros((config_in.shape[0], self.shape[0] * self.shape[1])).cuda()
        cnt = 0
        for i in range(4):
            config[:, self.shape[1] * 1 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, self.shape[1] * 2 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 3 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 4 + 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 5 + 0 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 6 + 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 7 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 8 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 9 + 2 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 10 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(12):
            config[:, self.shape[1] * 11 + 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(12):
            if i % 2 == 0:
                config[:, self.shape[1] * 12 + 4 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(8):
            config[:, self.shape[1] * 13 + 6 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, self.shape[1] * 14 + 8 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(4):
            config[:, self.shape[1] * 1 + 9 + i] = config_in[:, cnt]
            cnt += 1




        config = config.view(-1, 1, self.shape[0], self.shape[1])
        # config = config.view(-1, 1, 4, 4)
        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_SE_Kagome_12(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D_SE_Kagome_12, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.shape = (4,4)

        self.conv_list = nn.ModuleList()
        self.first_conv = Kagome12Conv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlockKagome12, Kagome12Conv2D, filters, num_blocks=2, height=self.shape, stride=1)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0]*self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config_in = config_in.clone().float()

        # config = torch.zeros((config_in.shape[0], 48)).cuda()
        # cnt = 0
        # for i in range(8):
        #     if i % 2 == 0:
        #         for j in range(6):
        #             if j % 2 == 0:
        #                 config[:, i * 6 + j] = config_in[:, cnt]
        #                 cnt += 1
        #     else:
        #         for j in range(6):
        #             config[:, i * 6 + j] = config_in[:, cnt]
        #             cnt += 1

        #2x2x3 kagome input
        config = torch.zeros((config_in.shape[0], 16)).cuda()
        cnt = 0
        for i in range(4):
            if i % 2 == 0:
                for j in range(4):
                    if j % 2 == 0:
                        config[:, i * 4 + j] = config_in[:, cnt]
                        cnt += 1
            else:
                for j in range(4):
                    config[:, i * 4 + j] = config_in[:, cnt]
                    cnt += 1

        # config = torch.zeros((config_in.shape[0], 64)).cuda()
        # cnt = 0
        # for i in range(4):
        #     if i % 2 == 0:
        #         config[:, 8 * 0 + i] = config_in[:, cnt]
        #         cnt += 1
        # for i in range(5):
        #     config[:, 8 * 1 + i] = config_in[:, cnt]
        #     cnt += 1
        # for i in range(6):
        #     if i % 2 == 0:
        #         config[:, 8 * 2 + i] = config_in[:, cnt]
        #         cnt += 1
        # for i in range(7):
        #     config[:, 8 * 3 + i] = config_in[:, cnt]
        #     cnt += 1
        # for i in range(8):
        #     if i % 2 == 0:
        #         config[:, 8 * 4 + i] = config_in[:, cnt]
        #         cnt += 1
        # for i in range(7):
        #     config[:, 8 * 5 + 1 + i] = config_in[:, cnt]
        #     cnt += 1
        # for i in range(6):
        #     if i % 2 == 0:
        #         config[:, 8 * 6 + 2 + i] = config_in[:, cnt]
        #         cnt += 1
        # for i in range(5):
        #     config[:, 8 * 7 + 3 + i] = config_in[:, cnt]
        #     cnt += 1


        # config = config.view(-1, 1, 8, 8)
        config = config.view(-1, 1, 4, 4)
        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_SE_Kagome(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, non_local=False, mode='embedded', preact=False):
        super(CNN2D_SE_Kagome, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.preact = preact
        self.shape = (8,8)

        self.conv_list = nn.ModuleList()
        self.first_conv = KagomeConv2D(self.filters_in, self.filters, kernel_size=4, stride=2, padding=1, padding_mode='circular')

        self.layer1 = self.make_layer(SEResidualBlockKagome, KagomeConv2D, filters, num_blocks=2, height=self.shape, stride=1, non_local=non_local, mode=mode, preact=preact)
        # self.layer2 = self.make_layer(SEResidualBlock, HexagonConv2d, filters, num_blocks=2, height=self.height, stride=1)

        self.linear1 = nn.Linear(self.shape[0] * self.shape[1] * self.filters, self.shape[0] * self.shape[1])
        self.linear2 = nn.Linear(self.shape[0] * self.shape[1], 2)

        self.first_In = nn.LayerNorm([self.filters, self.shape[0], self.shape[1]])

        self.Ln = nn.LayerNorm([self.shape[0]*self.shape[1]])

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, conv, filters, num_blocks, height, stride, non_local=False, mode='embedded', preact=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv, self.filters, filters, height, stride, preact))
            self.filters = filters
            if non_local:
                layers.append(NonLocalBlock(nn.Conv2d, self.filters, self.filters // 2, height, stride, mode=mode))
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        # batch, x, edge_index = data.batch, data.x, data.edge_index
        config_in = config_in.clone().float()

        # config = torch.zeros((config_in.shape[0], 48)).cuda()
        # cnt = 0
        # for i in range(8):
        #     if i % 2 == 0:
        #         for j in range(6):
        #             if j % 2 == 0:
        #                 config[:, i * 6 + j] = config_in[:, cnt]
        #                 cnt += 1
        #     else:
        #         for j in range(6):
        #             config[:, i * 6 + j] = config_in[:, cnt]
        #             cnt += 1

        # 2x2x3 kagome input
        # config = torch.zeros((config_in.shape[0], 16)).cuda()
        # cnt = 0
        # for i in range(4):
        #     if i % 2 == 0:
        #         for j in range(4):
        #             if j % 2 == 0:
        #                 config[:, i * 4 + j] = config_in[:, cnt]
        #                 cnt += 1
        #     else:
        #         for j in range(4):
        #             config[:, i * 4 + j] = config_in[:, cnt]
        #             cnt += 1

        config = torch.zeros((config_in.shape[0], 64)).cuda()
        cnt = 0
        for i in range(4):
            if i % 2 == 0:
                config[:, 8 * 0 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(5):
            config[:, 8 * 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, 8 * 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(7):
            config[:, 8 * 3 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(8):
            if i % 2 == 0:
                config[:, 8 * 4 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(7):
            config[:, 8 * 5 + 1 + i] = config_in[:, cnt]
            cnt += 1
        for i in range(6):
            if i % 2 == 0:
                config[:, 8 * 6 + 2 + i] = config_in[:, cnt]
                cnt += 1
        for i in range(5):
            config[:, 8 * 7 + 3 + i] = config_in[:, cnt]
            cnt += 1


        config = config.view(-1, 1, self.shape[0], self.shape[1])
        # config = config.view(-1, 1, 4, 4)
        config = self.first_conv(config)
        if not self.preact:
            config = self.first_In(config)
            config = F.relu(config)

        config = self.layer1(config)
        # config = self.layer2(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)

        config = self.Ln(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return self.log_psi, self.arg_psi

    def psi_batch(self, data, config):
        return self.psi(data, config)


class SEResidualBlockKagome(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, preact=False):
        super(SEResidualBlockKagome, self).__init__()

        self.preact = preact

        if self.preact:
            self.conv1 = nn.Sequential(
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
            )
            self.conv2 = nn.Sequential(
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            )
        else:
            self.conv1 = nn.Sequential(
                conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                )
            self.conv2 = nn.Sequential(
                conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                nn.LayerNorm([filters, height[0], height[1]]),
                # nn.ReLU(inplace=True),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # Squeeze
        w = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)
        if self.preact:
            return out
        else:
            out = self.relu(out)

        return out

class SEResidualBlockKagome12(torch.nn.Module):
    def __init__(self, conv, filters_in, filters, height, stride=1, preact=False):
        super(SEResidualBlockKagome12, self).__init__()

        self.preact = preact

        if self.preact:
            self.conv1 = nn.Sequential(
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
            )
            self.conv2 = nn.Sequential(
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            )
        else:
            self.conv1 = nn.Sequential(
                conv(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='circular'),
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
                )
            self.conv2 = nn.Sequential(
                conv(filters, filters, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
                nn.LayerNorm([filters, height[0], height[1]]),
                nn.ReLU(inplace=True),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

        self.fc1 = nn.Conv2d(filters, filters // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # Squeeze
        w = F.avg_pool2d(out, (out.shape[2], out.shape[3]))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out = out + self.shortcut(x)
        # out = F.relu(out)
        # if self.preact:
        #     return out
        # else:
        #     out = self.relu(out)

        return out

class CNN2D_block(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=3, kernel_size=3):
        super(CNN2D_block, self).__init__()
        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=kernel_size, stride=1, padding=1)
        for i in range(2):
            self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=kernel_size, stride=1, padding=1))

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 1)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
        self.In_list = nn.ModuleList()
        for i in range(2):
            self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))

    def forward(self, x):

        x = self.first_conv(x)
        x = self.first_In(x)
        x = F.relu(x)

        first_out = x
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            x = self.In_list[i](x)
            x = F.relu(x)

        x = x + first_out

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        # config = self.bn3(config)
        x = F.relu(x)
        # config = F.sigmoid(config)
        out = self.linear2(x)  # output log(|psi|) and arg(psi)

        return out

class CNN2D2(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
        super(CNN2D2, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.height = int(np.sqrt(self.num_visible))

        self.cnn_path1 = CNN2D_block(num_visible, num_hidden, filters_in=filters_in, filters=filters, kernel_size=3)
        self.cnn_path2 = CNN2D_block(num_visible, num_hidden, filters_in=filters_in, filters=filters, kernel_size=3)

        self.log_psi = 0
        self.arg_psi = 0

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        out1 = self.cnn_path1(config)
        out2 = self.cnn_path2(config)

        self.log_psi = out1 + 0 * out2
        self.arg_psi = out2 + 0 * out1

        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_sublattice(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3, device=torch.device('cpu')):
        super(CNN2D_sublattice, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.conv_list = nn.ModuleList()
        self.first_conv = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
        for i in range(2):
            self.conv_list.append(nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1))

        self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
        self.linear2 = nn.Linear(self.num_visible, 2)

        self.first_In = nn.LayerNorm([self.filters, self.height, self.height])
        self.In_list = nn.ModuleList()
        for i in range(2):
            self.In_list.append(nn.LayerNorm([self.filters, self.height, self.height]))

        self.log_psi = 0
        self.arg_psi = 0

        a = []
        a.append([1, 0] * 3)
        a.append([0, 1] * 3)
        b = np.array(a)
        c = np.tile(b, (3, 1))
        d = c.copy()
        mask_1 = np.where(d == 1)
        mask_0 = np.where(d == 0)
        d[mask_1] = 0
        d[mask_0] = 1
        e = np.stack((c, d))

        self.sublattice = torch.from_numpy(e).to(device)

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)
        sublattice = self.sublattice.repeat(config.shape[0], 1, 1, 1)
        config = torch.cat((config, sublattice), dim=1)

        config = self.first_conv(config)
        config = self.first_In(config)
        config = F.relu(config)

        first_out = config
        for i in range(len(self.conv_list)):
            config = self.conv_list[i](config)
            # if (i+1) != len(self.conv_list):
            config = self.In_list[i](config)
            config = F.relu(config)
        # config = self.conv_list[0](config)
        # config = F.relu(config)
        # config = self.conv_list[1](config)

        config = config + first_out
        # config = F.relu(config)

        config = config.view(config.size(0), -1)
        config = self.linear1(config)
        # config = self.bn3(config)
        config = F.relu(config)
        # config = F.sigmoid(config)
        out = self.linear2(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class CNN2D_v2(torch.nn.Module):
    def __init__(self, num_visible, num_hidden, filters_in=1, filters=16, kernel_size=3):
        super(CNN2D_v2, self).__init__()

        self.num_visible = num_visible
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters = filters
        self.height = int(np.sqrt(self.num_visible))

        self.first_conv = nn.Sequential(
            nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([filters, self.height, self.height]),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, filters=16, num_blocks=2, height=self.height, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, filters=32, num_blocks=2, height=self.height, stride=1)
        # self.layer3 = self.make_layer(ResidualBlock, filters=64, num_blocks=2, height=self.height, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(32, 2)
        # self.linear2 = nn.Linear(self.num_visible, 2)

        self.log_psi = 0
        self.arg_psi = 0

    def make_layer(self, block, filters, num_blocks, height, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.filters, filters, height, stride))
            self.filters = filters
        return nn.Sequential(*layers)

    def psi(self, data, config_in):
        batch, x, edge_index = data.batch, data.x, data.edge_index
        config = config_in.clone().float()
        config = config.view(-1, 1, self.height, self.height)

        config = self.first_conv(config)

        config = self.layer1(config)
        config = self.layer2(config)
        # config = self.layer3(config)

        config = self.avgpool(config)
        config = config.view(config.size(0), -1)
        out = self.linear1(config)  # output log(|psi|) and arg(psi)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]
        psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        return psi_value

    def psi_batch(self, data, config):
        return self.psi(data, config)

class ResidualBlock(torch.nn.Module):
    def __init__(self, filters_in, filters, height, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(filters_in, filters, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LayerNorm([filters, height, height]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LayerNorm([filters, height, height])
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or filters_in != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters_in, filters, kernel_size=1, stride=stride, padding=0, bias=True),
                nn.LayerNorm([filters, height, height])
            )

    def forward(self, x):
        out = self.conv(x)
        # print(out.shape)
        # print(x.shape)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

# class CNN2D(torch.nn.Module):
#     def __init__(self, num_visible, num_hidden, filters_in=1, filters=1, kernel_size=3):
#         super(CNN2D, self).__init__()
#
#         self.num_visible = num_visible
#         self.kernel_size = kernel_size
#         self.filters_in = filters_in
#         self.filters = filters
#         self.height = int(np.sqrt(self.num_visible))
#
#         self.conv0 = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
#         self.conv1 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
#         # self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
#         # self.linear2 = nn.Linear(self.num_visible, 2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(self.filters, 2)
#
#         self.bn0 = nn.BatchNorm2d(self.filters, eps=1e-5, momentum=0.997)
#         self.bn1 = nn.BatchNorm2d(self.filters, eps=1e-5, momentum=0.997)
#         self.bn2 = nn.BatchNorm2d(self.filters, eps=1e-5, momentum=0.997)
#         # self.bn3 = nn.BatchNorm1d(self.num_visible * self.filters, eps=1e-5, momentum=0.997)
#         # self.bn3 = nn.LayerNorm()
#
#         self.log_psi = 0
#         self.arg_psi = 0
#
#     def psi(self, data, config_in):
#         batch, x, edge_index = data.batch, data.x, data.edge_index
#         config = config_in.clone().float()
#         config = config.view(-1, 1, self.height, self.height)
#
#         config = self.conv0(config)
#         # config = self.bn0(config)
#         config = F.relu(config)
#
#         first_out = config
#         config = self.conv1(config)
#         # config = self.bn1(config)
#         config = F.relu(config)
#
#         config = self.conv2(config)
#         # config = self.bn2(config)
#         config = F.relu(config)
#
#         config = config + first_out
#
#         # config = config.view(config.size(0), -1)
#         # config = self.linear1(config)
#         # # config = self.bn3(config)
#         # config = F.relu(config)
#         # out = self.linear2(config)  # output log(|psi|) and arg(psi)
#
#         config = self.avgpool(config)
#         config = config.view(config.size(0), -1)
#         out = self.fc(config)
#
#         self.log_psi = out[:, 0:1]
#         self.arg_psi = out[:, 1:]
#         psi_value = (self.log_psi + 1j * self.arg_psi).exp()
#         return psi_value
#
#     def psi_batch(self, data, config):
#         return self.psi(data, config)

# class CNN2D_sublattice(torch.nn.Module):
#     def __init__(self, num_visible, num_hidden, filters_in=3, filters=1, kernel_size=3, device=torch.device('cpu')):
#         super(CNN2D_sublattice, self).__init__()
#
#         self.num_visible = num_visible
#         self.kernel_size = kernel_size
#         self.filters_in = filters_in
#         self.filters = filters
#         self.height = int(np.sqrt(self.num_visible))
#
#         self.conv1 = nn.Conv2d(self.filters_in, self.filters, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
#         self.linear1 = nn.Linear(self.num_visible * self.filters, self.num_visible)
#         self.linear2 = nn.Linear(self.num_visible, 2)
#
#         # self.bn1 = nn.BatchNorm2d(self.filters, eps=1e-5, momentum=0.997)
#         # self.bn2 = nn.BatchNorm2d(self.filters, eps=1e-5, momentum=0.997)
#         # self.bn3 = nn.BatchNorm1d(self.num_visible * self.filters, eps=1e-5, momentum=0.997)
#         # self.bn3 = nn.LayerNorm()
#
#         self.log_psi = 0
#         self.arg_psi = 0
#
#         a = []
#         a.append([1, 0] * 3)
#         a.append([0, 1] * 3)
#         b = np.array(a)
#         c = np.tile(b, (3, 1))
#         d = c.copy()
#         mask_1 = np.where(d == 1)
#         mask_0 = np.where(d == 0)
#         d[mask_1] = 0
#         d[mask_0] = 1
#         e = np.stack((c, d))
#
#         self.sublattice = torch.from_numpy(e).to(device)
#
#     def psi(self, data, config_in):
#         batch, x, edge_index = data.batch, data.x, data.edge_index
#         config = config_in.clone().float()
#         config = config.view(-1, 1, self.height, self.height)
#         sublattice = self.sublattice.repeat(config.shape[0], 1, 1, 1)
#         config = torch.cat((config, sublattice), dim=1)
#
#         config = self.conv1(config)
#         # config = self.bn1(config)
#         config = F.relu(config)
#
#         config = self.conv2(config)
#         # config = self.bn2(config)
#         config = F.relu(config)
#
#         config = config.view(config.size(0), -1)
#         config = self.linear1(config)
#         # config = self.bn3(config)
#         config = F.relu(config)
#         out = self.linear2(config)  # output log(|psi|) and arg(psi)
#
#         self.log_psi = out[:, 0:1]
#         self.arg_psi = out[:, 1:]
#         psi_value = (self.log_psi + 1j * self.arg_psi).exp()
#         return psi_value
#
#     def psi_batch(self, data, config):
#         return self.psi(data, config)

class MLP3layer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
        super(MLP3layer, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.activation = activation
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)   
        self.ln3 = torch.nn.LayerNorm(hidden_dim)   
#         self.bn = torch.nn.BatchNorm1d(hidden_dim)
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.ln3.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = self.ln1(x)
        x = self.activation(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.ln2(x) 
        x = self.activation(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        x = self.ln3(x) 
        x = self.activation(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin4(x)

        return x

# class MLP3layer(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
#         super(MLP3layer, self).__init__()
#         self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.lin4 = torch.nn.Linear(hidden_dim, output_dim)
#         self.dropout = dropout
#         self.activation = activation
#         self.ln1 = torch.nn.LayerNorm(input_dim)
#         self.ln2 = torch.nn.LayerNorm(hidden_dim)   
#         self.ln3 = torch.nn.LayerNorm(hidden_dim)   
#         self.ln4 = torch.nn.LayerNorm(hidden_dim) 
# #         self.bn = torch.nn.BatchNorm1d(hidden_dim)
        
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()
#         self.lin4.reset_parameters()
#         self.ln1.reset_parameters()
#         self.ln2.reset_parameters()
#         self.ln3.reset_parameters()
#         self.ln4.reset_parameters()

#     def forward(self, x):
#         x = self.ln1(x)
#         x = self.activation(x)
#         x = self.lin1(x)  

#         x = self.ln2(x)
#         x = self.activation(x)
#         x = self.lin2(x)
        
#         x = self.ln3(x) 
#         x = self.activation(x)
#         x = self.lin3(x)
        
#         x = self.ln4(x) 
#         x = self.activation(x)
#         x = self.lin4(x)
#         return x


class EdgeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP3layer(input_dim, hidden_dim, output_dim, dropout, F.relu)

    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, input_dim1, hidden_dim1, output_dim1, input_dim2, hidden_dim2, output_dim2, dropout):
        super(NodeModel, self).__init__()
        # self.node_mlp_1 = MLP3layer(input_dim1, hidden_dim1, output_dim1, dropout, F.relu)
        self.node_mlp_2 = MLP3layer(input_dim2, hidden_dim2, output_dim2, dropout, F.relu)

    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        # out = torch.cat([x[row], edge_attr], dim=1)
        # out = self.node_mlp_1(out)
        out = edge_attr
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class CustomMetaLayer(torch.nn.Module):
    def __init__(self, edge_input_dim, edge_hidden_dim, edge_output_dim, node_input_dim, node_hidden_dim,
                 node_output_dim, dropout):
        super(CustomMetaLayer, self).__init__()


        self.edge_model = EdgeModel(node_input_dim * 2 + edge_input_dim, edge_hidden_dim, edge_output_dim, dropout)
        self.node_model = NodeModel(node_input_dim + edge_output_dim, node_hidden_dim, edge_output_dim,
                                    edge_output_dim + node_input_dim, node_hidden_dim, node_output_dim, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):

        updated_edge_attr = self.edge_model(x, edge_index, edge_attr, batch)
        updated_x = self.node_model(x, edge_index, updated_edge_attr + edge_attr, batch)

        return x + updated_x, edge_attr + updated_edge_attr

def sublatticeEncoding(edges, shape = 'square', num_nodes = 100, classes = 2):
    from collections import defaultdict
    neighbour = defaultdict(list)
    for i in range(edges.shape[1]):
        u,v = edges[:,i]
        neighbour[int(u)].append(int(v))

    if shape == "square":
        encoding = {-1:0}

        q = [0]
        parent = {0:-1}
        visited = [False] * num_nodes
        while len(q) > 0:
            node = q.pop()
            visited[node] = True
            encoding[node] = (encoding[parent[node]] + 1) % classes

            for i in neighbour[node]:
                if not visited[i]:
                    q.append(i)
                    parent[i] = node

        ret = torch.tensor([encoding[i] for i in range(num_nodes)])
        return torch.nn.functional.one_hot(ret, num_classes = classes) 

    elif shape == "kag":
        # encoding = {0:0}
        # visited = [False] * num_nodes
        # visited[0] = True
        # q = []
        # local_neigh = sorted(neighbour[0])
        # encoding[local_neigh[0]] = 1
        # encoding[local_neigh[1]] = 2
        # q.append(local_neigh[0])
        # q.append(local_neigh[1])

        # print(q)  
        # while len(q) > 0:
        #     node = q.pop(0)
        #     visited[node] = True
            
        #     local_neigh = sorted(neighbour[node])
            
        #     for i in range(4):
        #         if (local_neigh[i] not in encoding.keys()):
        #             encoding[local_neigh[i]] = encoding[local_neigh[3-i]]
        #         if not visited[local_neigh[i]] and (local_neigh[i] not in q):
        #             q.append(local_neigh[i])
        #     print(node)
        #     print(q)
        #     print(encoding)
        if (num_nodes == 36):
            encoding = {0:0, 1:0, 2:1, 3:2, 4:1, 5:2, 6:1, 7:0, 8:0, 9:0, 10:1, 11:2, 12:1, 13:2, 14:1, 15:2, 16:1, 17:0, 18:0, 19:0, 20:0, 21:2, 22:1, 23:2, 24:1, 25:2, 26:1, 27:2, 28:0, 29:0, 30:0, 31:2, 32:1, 33:2, 34:1, 35:2}

            ret = torch.tensor([encoding[i] for i in range(num_nodes)])
            return torch.nn.functional.one_hot(ret, num_classes = classes) 
        elif (num_nodes == 12):
            encoding = {0:0, 1:0, 2:1, 3:2, 4:1, 5:2, 6:0, 7:0, 8:1, 9:2, 10:1, 11:2}

            ret = torch.tensor([encoding[i] for i in range(num_nodes)])
            return torch.nn.functional.one_hot(ret, num_classes = classes) 
        else:
            raise Exception("invalid kagome node number")
    
    elif shape == 'triangle':
        if num_nodes == 36:
            encoding = {}
            l = int(np.sqrt(num_nodes))
            for i in range(l):
                c = i % classes
                for j in range(l):
                    encoding[ i * l + j] = c 
                    c = (c + 1) % classes

            ret = torch.tensor([encoding[i] for i in range(num_nodes)])
            return torch.nn.functional.one_hot(ret, num_classes = classes)           
        elif num_nodes == 108:
            encoding = {}
            idx = 0
            l = [8, 9, 9, 9, 10, 11, 10, 9, 9, 9, 8, 7]
            leading = [0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0, 2]
            for i in range(len(l)):
                c = leading[i]
                for j in range(l[i]):
                    encoding[idx] = c 
                    idx += 1
                    c = (c + 1) % classes

            ret = torch.tensor([encoding[i] for i in range(num_nodes)])
            return torch.nn.functional.one_hot(ret, num_classes = classes)
    
    elif shape == "honeycomb":
        if num_nodes == 32:
            a = []
            a.append([1] * 4)
            a.append([0] * 4)
            b = np.array(a)
            c = np.tile(b, (4, 1))
            c = c.reshape(-1)
            ret = torch.tensor(c)
            return torch.nn.functional.one_hot(ret, num_classes = classes) 

class GraphNet(torch.nn.Module):
    def __init__(self, num_visible, num_node_features, num_edge_features, dropout, depth, hidden=128, emb_dim = 64, device=torch.device('cpu'), sublattice = False, sublatticeCls = 2, shape = "square"):
        super(GraphNet, self).__init__()

        self.num_visible = num_visible
        self.device = device

        self.metalayer = torch.nn.ModuleList()
        self.metalayer.append(
            CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))
        self.depth = depth
        for i in range(self.depth - 1):
            self.metalayer.append(
                CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))

        # self.mlp = MLP3layer(emb_dim * 2, hidden, 2, dropout, F.relu)
        self.mlp = torch.nn.Linear(emb_dim * 2, 2)
            
        self.dropout = dropout
        # self.bn_node = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth + 1)])
        # self.bn_edge = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth + 1)])
        # self.bn_global = torch.nn.ModuleList([BatchNorm(hidden) for i in range(self.depth + 1)])
        # self.ln_node = nn.LayerNorm(emb_dim)
        # self.ln_edge = nn.LayerNorm(emb_dim)

        self.mlp_edge = MLP3layer(num_edge_features, hidden, emb_dim, 0, F.relu)
        # self.mlp_edge = torch.nn.Linear(num_edge_features, emb_dim)
        if sublattice:
            self.mlp_node = MLP3layer(num_node_features + sublatticeCls, hidden, emb_dim, dropout, F.relu)
            # self.mlp_node = torch.nn.Linear(num_node_features + sublatticeCls, emb_dim)
        else:
            self.mlp_node = MLP3layer(num_node_features, hidden, emb_dim, dropout, F.relu)
            # self.mlp_node = torch.nn.Linear(num_node_features, emb_dim)
        
        # self.decoder_node = torch.nn.Linear(emb_dim, emb_dim)
        # self.decoder_edge = torch.nn.Linear(emb_dim, emb_dim)
        self.decoder_node = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)
        self.decoder_edge = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)
        
        self.sublattice = sublattice

        self.SLencoding = None
        self.sublatticeCls = sublatticeCls

        self.batch_sample = None
        self.batch_energy = None
        self.batch_energy2 = None
        self.batch = []
        self.batch_sample_size = []

        self.shape = shape

    def psi(self, data, config_in):
        # print(data)
        # print(config_in.shape)
        if self.sublattice:
            if self.SLencoding == None:
                self.SLencoding = sublatticeEncoding(data.edge_index, num_nodes = config_in.shape[1], shape = self.shape, classes = self.sublatticeCls).to(self.device).float()
                # self.SLencoding = self.SLencoding.view(1,-1,self.sublatticeCls).repeat(config_in.shape[0],1,1).reshape(-1,self.sublatticeCls)

        if len(self.batch) == 0:
            self.batch.append(state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding))
        else:
            flag = False
            for b in self.batch:
                if b.batch.shape[0] == config_in.view(-1,1).shape[0]:
                    flag = True
            if not flag:        
                self.batch.append(state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding))
                
        # if self.batch_sample == None:
        #     self.batch_sample = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
        # elif (self.batch_energy == None) and self.batch_sample.batch.shape[0] != config_in.view(-1,1).shape[0]:
        #     self.batch_energy = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
        # elif self.batch_energy2 == None and self.batch_sample.batch.shape[0] != config_in.view(-1,1).shape[0] and self.batch_energy.batch.shape[0] != config_in.view(-1,1).shape[0]:
        #     self.batch_energy2 = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)

        # # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # print(config_in.shape)
        # print(self.SLencoding.shape)
        # import pdb; pdb.set_trace()

        x = 0
        if (self.sublattice):
            x = torch.cat((config_in.float().view(-1, 1), self.SLencoding.view(1,-1,self.sublatticeCls).repeat(config_in.shape[0],1,1).view(-1,self.sublatticeCls)), 1)
        else:
            x = config_in.float().view(-1, 1)

        for b in self.batch:
            if b.batch.shape[0] == config_in.view(-1,1).shape[0]:
                edge_index, edge_attr, batch = b.edge_index.clone(), b.edge_attr.clone(), b.batch.clone()
                break

        # if self.batch_sample.batch.shape[0] == config_in.view(-1,1).shape[0]:
        #     edge_index, edge_attr, batch = self.batch_sample.edge_index.clone(), self.batch_sample.edge_attr.clone(), self.batch_sample.batch.clone()
        # elif self.batch_energy.batch.shape[0] == config_in.view(-1,1).shape[0]:
        #     edge_index, edge_attr, batch = self.batch_energy.edge_index.clone(), self.batch_energy.edge_attr.clone(), self.batch_energy.batch.clone()
        # else:
        #     edge_index, edge_attr, batch = self.batch_energy2.edge_index.clone(), self.batch_energy2.edge_attr.clone(), self.batch_energy2.batch.clone()
            # batched_data = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
            # # print("sample size", self.batch_sample.batch.shape)
            # # print("energy size", self.batch_energy.batch.shape)
            # # print("config size", config_in.shape)
            # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # # print(self.SLencoding.shape)
        # batched_data = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)

        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.mlp_node(x)
        edge_attr = self.mlp_edge(edge_attr)

        z_arr = []

        for i in range(self.depth):
            x, edge_attr = self.metalayer[i](x, edge_index, edge_attr, batch)
            # x = self.bn_node[i](x)
            # edge_attr = self.bn_edge[i](edge_attr)    
        
        x = self.decoder_node(x)
        edge_attr = self.decoder_edge(edge_attr)

        # readout
        x_sum = global_add_pool(x, batch)
        edge_batch = batch[edge_index[0]]
        edge_sum = scatter_sum(edge_attr, edge_batch, dim=0)
        embed = torch.cat((x_sum, edge_sum), 1)

        # regression
        out = self.mlp(embed)

        self.log_psi = out[:, 0:1]
        self.arg_psi = out[:, 1:]

        return self.log_psi, self.arg_psi

        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        # return psi_value


    def psi_batch(self, data, config):
        return self.psi(data, config)

class GraphNet2(torch.nn.Module):
    def __init__(self, num_visible, num_node_features, num_edge_features, dropout, depth, hidden=128, emb_dim = 64, device=torch.device('cpu'), sublattice = True, sublatticeCls = 2):
        super(GraphNet2, self).__init__()

        self.num_visible = num_visible
        self.device = device

        ###  GNN 1
        self.metalayer = torch.nn.ModuleList()
        self.metalayer.append(
            CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))
        self.depth = depth
        for i in range(self.depth - 1):
            self.metalayer.append(
                CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))

        # self.mlp = MLP3layer(emb_dim * 2, hidden, 2, dropout, F.relu)
        self.mlp = torch.nn.Linear(emb_dim * 2, 1)
            
        self.dropout = dropout

        # self.mlp_edge = MLP3layer(num_edge_features, hidden, emb_dim, 0, F.relu)
        self.mlp_edge = torch.nn.Linear(num_edge_features, emb_dim)
        if sublattice:
            # self.mlp_node = MLP3layer(num_node_features + sublatticeCls, hidden, emb_dim, dropout, F.relu)
            self.mlp_node = torch.nn.Linear(num_node_features + sublatticeCls, emb_dim)
        else:
            # self.mlp_node = MLP3layer(num_node_features, hidden, emb_dim, dropout, F.relu)
            self.mlp_node = torch.nn.Linear(num_node_features, emb_dim)
        
        # self.decoder_node = torch.nn.Linear(emb_dim, emb_dim)
        # self.decoder_edge = torch.nn.Linear(emb_dim, emb_dim)
        self.decoder_node = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)
        self.decoder_edge = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)

        ## GNN2
        self.metalayer2 = torch.nn.ModuleList()
        self.metalayer2.append(
            CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))
        for i in range(self.depth - 1):
            self.metalayer2.append(
                CustomMetaLayer(emb_dim, hidden, emb_dim, emb_dim, hidden, emb_dim, dropout))

        # self.mlp = MLP3layer(emb_dim * 2, hidden, 2, dropout, F.relu)
        self.mlp2 = torch.nn.Linear(emb_dim * 2, 1)

        # self.mlp_edge = MLP3layer(num_edge_features, hidden, emb_dim, 0, F.relu)
        self.mlp_edge2 = torch.nn.Linear(num_edge_features, emb_dim)
        if sublattice:
            # self.mlp_node = MLP3layer(num_node_features + sublatticeCls, hidden, emb_dim, dropout, F.relu)
            self.mlp_node2 = torch.nn.Linear(num_node_features + sublatticeCls, emb_dim)
        else:
            # self.mlp_node = MLP3layer(num_node_features, hidden, emb_dim, dropout, F.relu)
            self.mlp_node2 = torch.nn.Linear(num_node_features, emb_dim)
        
        self.decoder_node2 = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)
        self.decoder_edge2 = MLP3layer(emb_dim, hidden, emb_dim, 0, F.relu)
        
        self.sublattice = sublattice

        self.SLencoding = None
        self.sublatticeCls = sublatticeCls

        self.batch_sample = None
        self.batch_energy = None
        self.batch_energy2 = None

    def psi(self, data, config_in):

        if self.sublattice:
            if self.SLencoding == None:
                self.SLencoding = sublatticeEncoding(data.edge_index, num_nodes = config_in.shape[1], shape = 'square', classes = self.sublatticeCls).to(self.device).float()
                # self.SLencoding = self.SLencoding.view(1,-1,self.sublatticeCls).repeat(config_in.shape[0],1,1).reshape(-1,self.sublatticeCls)

        if self.batch_sample == None:
            self.batch_sample = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
        elif (self.batch_energy == None) and self.batch_sample.batch.shape[0] != config_in.view(-1,1).shape[0]:
            self.batch_energy = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
        elif self.batch_energy2 == None and self.batch_sample.batch.shape[0] != config_in.view(-1,1).shape[0] and self.batch_energy.batch.shape[0] != config_in.view(-1,1).shape[0]:
            self.batch_energy2 = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)

        # # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # print(config_in.shape)
        # print(self.SLencoding.shape)
        x = 0
        if (self.sublattice):
            x = torch.cat((config_in.float().view(-1, 1), self.SLencoding.view(1,-1,self.sublatticeCls).repeat(config_in.shape[0],1,1).view(-1,self.sublatticeCls)), 1)
        else:
            x = config_in.float().view(-1, 1)

        if self.batch_sample.batch.shape[0] == config_in.view(-1,1).shape[0]:
            edge_index, edge_attr, batch = self.batch_sample.edge_index.clone(), self.batch_sample.edge_attr.clone(), self.batch_sample.batch.clone()
        elif self.batch_energy.batch.shape[0] == config_in.view(-1,1).shape[0]:
            edge_index, edge_attr, batch = self.batch_energy.edge_index.clone(), self.batch_energy.edge_attr.clone(), self.batch_energy.batch.clone()
        else:
            edge_index, edge_attr, batch = self.batch_energy2.edge_index.clone(), self.batch_energy2.edge_attr.clone(), self.batch_energy2.batch.clone()
            # batched_data = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)
            # # print("sample size", self.batch_sample.batch.shape)
            # # print("energy size", self.batch_energy.batch.shape)
            # # print("config size", config_in.shape)
            # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # # print(self.SLencoding.shape)
        # batched_data = state_to_pygdata(data, config_in, self.device, self.sublattice, self.SLencoding)

        # x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x2 = x.clone()
        edge_attr2 = edge_attr.clone()
        
        x2 = self.mlp_node2(x2)
        edge_attr2 = self.mlp_edge2(edge_attr2)

        for i in range(self.depth):
            x2, edge_attr2 = self.metalayer2[i](x2, edge_index, edge_attr2, batch)
            # x = self.bn_node[i](x)
            # edge_attr = self.bn_edge[i](edge_attr)    
        
        x2 = self.decoder_node2(x2)
        edge_attr2 = self.decoder_edge2(edge_attr2)

        # readout
        x2_sum = global_add_pool(x2, batch)
        edge_batch = batch[edge_index[0]]
        edge2_sum = scatter_sum(edge_attr2, edge_batch, dim=0)
        embed2 = torch.cat((x2_sum, edge2_sum), 1)

        # regression
        out = self.mlp2(embed2)
        self.arg_psi = out[:, 0:1]
        # self.arg_psi = self.mlp2(embed2)

        x = self.mlp_node(x)
        edge_attr = self.mlp_edge(edge_attr)

        for i in range(self.depth):
            x, edge_attr = self.metalayer[i](x, edge_index, edge_attr, batch)
            # x = self.bn_node[i](x)
            # edge_attr = self.bn_edge[i](edge_attr)    
        
        x = self.decoder_node(x)
        edge_attr = self.decoder_edge(edge_attr)

        # readout
        x_sum = global_add_pool(x, batch)
        edge_batch = batch[edge_index[0]]
        edge_sum = scatter_sum(edge_attr, edge_batch, dim=0)
        embed = torch.cat((x_sum, edge_sum), 1)

        # regression
        out = self.mlp(embed)
        self.log_psi = out[:, 0:1]
        # self.arg_psi = out[:, 1:]

        # self.log_psi = self.mlp(embed)

        return self.log_psi, self.arg_psi

        # psi_value = (self.log_psi + 1j * self.arg_psi).exp()
        # return psi_value


    def psi_batch(self, data, config):
        return self.psi(data, config)

def state_to_pygdata(data_in, state, device, sublattice, SLencoding = None):

    data_in['batch'] = None

    data_list = []

    for i in range(state.shape[0]):
        config = state[i]
        if sublattice:
            x = torch.cat((config.clone().float().view(-1, 1), SLencoding), 1)
        else:
            x = config.clone().float().view(-1, 1)
        data = Data(x = x, edge_index = data_in.clone().edge_index, edge_attr = torch.zeros((data_in.edge_index.shape[1], 1)).float() )
        data_list.append(data.to(device))

    batched_data = Batch.from_data_list(data_list)

    return batched_data



class FFN_twoway(torch.nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(FFN_twoway, self).__init__()
        
        self.num_visible = num_visible
        
        self.lin1 = torch.nn.Linear(num_visible, num_hidden)
        self.lin2 = torch.nn.Linear(num_hidden, 1)
        self.sign_mlp = nn.Linear(num_visible, 1)
#         self.dropout = dropout
        self.relu1 = ReLU()
#         self.bn = torch.nn.BatchNorm1d(hidden_dim)
        

#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()

    def psi(self, data, config):
#         x = F.dropout(x, p=self.dropout, training=self.training)
        config = config.float()    
        x = self.lin1(config)
#         x = self.bn(x)
        x = self.relu1(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        amplitude = torch.sigmoid(x)

        sign = (self.sign_mlp(config)).cos()
        
        return sign * amplitude
    
class FFN_twoway_sigmoid(torch.nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(FFN_twoway_sigmoid, self).__init__()
        
        self.num_visible = num_visible
        
        self.lin1 = torch.nn.Linear(num_visible, num_hidden)
        self.lin2 = torch.nn.Linear(num_hidden, 1)
        self.sign_mlp = nn.Linear(num_visible, 1)
#         self.dropout = dropout
#         self.relu = ReLU(inplace=True)
#         self.bn = torch.nn.BatchNorm1d(hidden_dim)
        

#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()

    def psi(self, data, config):
#         x = F.dropout(x, p=self.dropout, training=self.training)
        config = config.float()    
        x = self.lin1(config)
#         x = self.bn(x)
        x = torch.sigmoid(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        amplitude = torch.sigmoid(x)
        

        sign = (self.sign_mlp(config)).cos()
        
        return sign * amplitude

class RBM(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible).float() * 1e-1)
        self.v_bias = nn.Parameter(torch.randn(num_visible).float() * 1e-1)
        self.h_bias = nn.Parameter(torch.randn(num_hidden).float() * 1e-1)

        # self.a = nn.Parameter(torch.tensor([1.0]))
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
    def psi(self, data, v):
        '''
        probability for visible nodes, visible/hidden nodes here take value in {-1, 1}.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v = v.float()
        if self.W.is_cuda: v = v.cuda()
        res = (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()
        return res
    
class RBM_twoway(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_hidden):
        super(RBM_twoway, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible).float() * 1e-1)
        self.v_bias = nn.Parameter(torch.randn(num_visible).float() * 1e-1)
        self.h_bias = nn.Parameter(torch.randn(num_hidden).float() * 1e-1)

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        self.sign_mlp = nn.Linear(num_visible, 1)
        
    def psi(self, data, v):
        '''
        probability for visible nodes, visible/hidden nodes here take value in {-1, 1}.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v = v.float()
        if self.W.is_cuda: v = v.cuda()
        res = (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()
        
        sign = (self.sign_mlp(v)).cos()
        
        return res*sign
    
class Model_twoway_invariant(nn.Module):

    def __init__(self, data, num_visible, num_hidden, model):
        super(Model_twoway_invariant, self).__init__()
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        if model == 'RBM-twoway-invariant':
            self.model = RBM_twoway(num_visible, num_hidden)
        elif model == 'RBM-mul-twoway-invariant':
            self.model = RBM_mul_twoway(num_visible, num_hidden)
        elif model == 'cnn1d-twoway-invariant':
            self.model = CNN1D_twoway(num_visible, num_hidden, filter_num=4, kernel_size=3)
        elif model == 'gnn-transformer-twoway-invariant':
            self.model = GT_twoway(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif model == 'gnn-transformer-invariant':
            self.model = GT(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif model == 'deepergcn-virtual-twoway-invariant':
            self.model = DeeperGCN_Virtualnode_twoway(data.num_nodes,
                                   num_layers=5, 
                                   emb_dim=32, 
                                   drop_ratio = 0, 
                                   JK = 'last', 
                                   graph_pooling='sum', 
                                   norm='layer')
        
    def psi(self, data, v):

        v = v.float()
        v_bar = v.clone() * -1
        
        res = self.model.psi(data, v)
        res_bar = self.model.psi(data, v_bar)
        
        res = (res + res_bar) / 2
        
        return res

class Model_twoway_invariant_v2(nn.Module):

    def __init__(self, data, num_visible, num_hidden, model):
        super(Model_twoway_invariant_v2, self).__init__()
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        if model == 'RBM-twoway-invariant-v2':
            self.model = RBM_twoway(num_visible, num_hidden)
        elif model == 'RBM-mul-twoway-invariant-v2':
            self.model = RBM_mul_twoway(num_visible, num_hidden)
        elif model == 'gnn-transformer-twoway-invariant-v2':
            self.model = GT_twoway(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif model == 'gnn-transformer-invariant-v2':
            self.model = GT(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        
    def psi(self, data, v):

        v = v.float()
        v_bar = v.clone() * -1
        
        res = self.model.psi(data, v)
        res_bar = self.model.psi(data, v_bar)
        
        res = (0.5 * ((2*res).exp() + (2*res_bar).exp())).log() * 0.5
        
        return res

class RBM_mul_twoway(nn.Module):

    def __init__(self, num_visible, num_hidden):
        super(RBM_mul_twoway, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible).float() * 1e-1)
        self.v_bias = nn.Parameter(torch.randn(num_visible).float() * 1e-1)
        self.h_bias = nn.Parameter(torch.randn(num_hidden).float() * 1e-1)
        
        # polynomial part weights
        self.d_bias = nn.Parameter(torch.randn(1).float() * 1e-1)
        self.c = nn.Parameter(torch.randn(num_visible).float() * 1e-1)
        self.A = nn.Parameter(torch.randn(num_visible, num_visible).float() * 1e-1)

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        self.sign_mlp = nn.Linear(num_visible, 1)
        
    def psi(self, data, v):
        '''
        probability for visible nodes, visible/hidden nodes here take value in {-1, 1}.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v = v.float()
        if self.W.is_cuda: v = v.cuda()
        res = (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp() * (self.d_bias + self.c.dot(v) + v.dot(self.A.mv(v)))
        sign = (self.sign_mlp(v)).cos()
        
        return res*sign
    
class CNN1D(nn.Module):
    def __init__(self, num_visible, num_hidden, filter_num=4, kernel_size=3):
        super(CNN1D, self).__init__()
        self.filter_num = filter_num
        self.conv_padding = 1
        self.conv_stride = 1
        self.pooling_stride = 2
        self.conv = nn.Conv1d(1, self.filter_num, 3, stride=1)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        
        out_size = (num_visible + 2 * self.conv_padding - kernel_size) / self.conv_stride + 1
        out_size = (out_size - 2) / self.pooling_stride + 1
        
        self.deconv_size = int(num_visible / out_size)
        
        self.d = nn.Parameter(torch.randn(self.deconv_size,self.filter_num) * 1e-1)
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        
    def psi(self, data, v_in):
        v = v_in
        v = v.float().view(1,1,-1)
        v = F.pad(v, (1,1), mode='circular')
        v = self.conv(v)
        v = self.maxpool(v)
        
        v = v.squeeze(0).T
        c = []
        for i in range(self.d.shape[1]):
            tmp = torch.kron(v[:,i,None],self.d[:,i,None])
            c.append(tmp)
        res = torch.cat(c,axis=1)
        res = torch.sum(res, axis=1).prod()
        
        return res

class CNN1D_twoway(nn.Module):
    def __init__(self, num_visible, num_hidden, filter_num=4, kernel_size=3):
        super(CNN1D_twoway, self).__init__()
        self.filter_num = filter_num
        self.conv_padding = 1
        self.conv_stride = 1
        self.pooling_stride = 2
        self.conv = nn.Conv1d(1, self.filter_num, 3, stride=1)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        
        out_size = (num_visible + 2 * self.conv_padding - kernel_size) / self.conv_stride + 1
        out_size = (out_size - 2) / self.pooling_stride + 1
        
        self.deconv_size = int(num_visible / out_size)
        
        self.d = nn.Parameter(torch.randn(self.deconv_size,self.filter_num) * 1e-1)
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        self.sign_mlp = nn.Linear(num_visible, 1)

    def psi(self, data, v_in):
        v = v_in
        v = v.float().view(1,1,-1)
        v = F.pad(v, (1,1), mode='circular')
        v = self.conv(v)
        v = self.maxpool(v)
        
        v = v.squeeze(0).T
        c = []
        for i in range(self.d.shape[1]):
            tmp = torch.kron(v[:,i,None],self.d[:,i,None])
            c.append(tmp)
        res = torch.cat(c,axis=1)
        res = torch.sum(res, axis=1).prod()
        
        sign = (self.sign_mlp(v_in.float())).cos()
        
        return res * sign
    
    def psi_batch(self, data, v_in):
        v = v_in
        v = v.float().unsqueeze(1)
        v = F.pad(v, (1,1), mode='circular')
        v = self.conv(v)
        v = self.maxpool(v)
        
        # v = v.squeeze(0).T
        v = v.transpose(1, 2)
        c = []
        for i in range(self.d.shape[1]):
            tmp = torch.kron(v[:,:,i,None],self.d[None,:,i,None])
            c.append(tmp)
        res = torch.cat(c,axis=2)
        res = torch.sum(res, axis=2).prod(dim=-1, keepdim=True)
        
        sign = (self.sign_mlp(v_in.float())).cos()
        return res * sign

class SizeNorm(torch.nn.Module):
    def __init__(self):
        super(SizeNorm, self).__init__()

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        inv_sqrt_deg = degree(batch, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg[batch].view(-1, 1)    

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
    
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = torch.nn.Linear(hidden_size, ffn_size)
        self.gelu = torch.nn.GELU()
#         self.relu = ReLU(inplace=True)
#         self.swish = Swish()
        self.layer2 = torch.nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
#         x = self.relu(x)
#         x = self.swish(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = torch.nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = torch.nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = torch.nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = torch.nn.Dropout(attention_dropout_rate)

        self.output_layer = torch.nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, batch_idx):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.head_size, d_k)
        k = self.linear_k(k).view(-1, self.head_size, d_k)
        v = self.linear_v(v).view(-1, self.head_size, d_v)

        q = q.transpose(0, 1)                  # [h, q_len, d_k]
        v = v.transpose(0, 1)                  # [h, v_len, d_v]
        k = k.transpose(0, 1).transpose(1, 2)  # [h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [h, q_len, k_len]
        
        

#         ### Softmax attention scores should be computed within each graph seperately in the batch
#         total_nodes_batch = orig_q_size[0] # Total number of nodes in the batch
#         x_exp = torch.exp(x) # [h, q_len, k_len]
#         norm_const_mx = scatter_sum(x_exp, batch_idx, dim=-1) # [h, q_len, num_graphs]
#         norm_const = norm_const_mx[:, range(total_nodes_batch), batch_idx] # [h, q_len]
#         attn_mx = x_exp/norm_const.unsqueeze(1).transpose(1,2) # [h, q_len, k_len]
#         # Mask attention scores across different graphs
#         n_nodes = scatter_sum(torch.ones(total_nodes_batch, device=q.device), batch_idx, dim=-1) # [num_graphs, ]
#         mask = torch.block_diag(*[torch.ones(int(n_node),int(n_node), device=q.device) for n_node in n_nodes]) # [q_len, q_len]
#         x = attn_mx * mask # [h, q_len, q_len]
        
        x = torch.softmax(x, dim=2)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [h, q_len, d_v]

        x = x.transpose(0, 1).contiguous()  # [q_len, h, d_v]
        x = x.view(-1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x    
    
    
class TranLayer(torch.nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(TranLayer, self).__init__()

        self.self_attention_norm = torch.nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = torch.nn.Dropout(dropout_rate)

        self.ffn_norm = torch.nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, batch_idx):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, batch_idx)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    
    
class GT_Sym(torch.nn.Module):
    def __init__(self, num_nodes, num_layers_gnn, num_layers_tran, emb_dim, head_size, dropout_rate = 0, attention_dropout_rate = 0, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):

        super(GT_Sym, self).__init__()
        
        self.num_visible = num_nodes
        
        self.JK = JK
        
        self.node_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        

        ### List of GNNs
        self.first_gnn_conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, num_layers_gnn + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
#             act = torch.nn.GELU()
#             act = Swish()


            gnn_layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.gnn_layers.append(gnn_layer)

           
        ### List of Transformer Layer
        self.tran_layers = torch.nn.ModuleList()
        for k in range(1, num_layers_tran + 1):
            tran_layer = TranLayer(emb_dim, emb_dim, dropout_rate, attention_dropout_rate, head_size)
            self.tran_layers.append(tran_layer)
            
            
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
#         self.mlp_sign = torch.nn.Linear(emb_dim, num_tasks)
        

    def psi(self, config_in, perturb=None):
#         print('Inside Model:  num graphs: {}, device: {}'.format(batched_data.num_graphs, batched_data.batch.device))

        if config_in[0] == -1:
            config = -config_in
        else:
            config = config_in
    
        data.x = config.to(torch.float32).view(-1,1)
    
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x) + perturb if perturb is not None else self.node_mlp(x)
        h = self.node_mlp(x)

        h = self.first_gnn_conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = self.tran_layers[i](h, batch)
            h = gnn_layer(h, edge_index, edge_attr)
            h_list.append(h)
        
        
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        ## two way
#         amplitude = torch.sigmoid(output)
        
#         sign_res = self.mlp_sign(h_graph)
        
#         output = amplitude * torch.cos(sign_res)
       
#         if config_in[0] == -1:
#             return -output
#         else:
        return output

class GT_sublattice(torch.nn.Module):
    def __init__(self, num_nodes, num_layers_gnn, num_layers_tran, emb_dim, head_size, dropout_rate = 0, attention_dropout_rate = 0, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):

        super(GT_sublattice, self).__init__()
        
        self.num_visible = num_nodes
        
        self.JK = JK
        
        self.node_mlp = MLP(3, emb_dim, emb_dim, attention_dropout_rate, F.relu)
#         self.SpinEncoder = SpinEncoder(emb_dim)
        
        self.edge_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        

        ### List of GNNs
        self.first_gnn_conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, num_layers_gnn + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
#             act = torch.nn.GELU()
#             act = Swish()


            gnn_layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.gnn_layers.append(gnn_layer)

           
        ### List of Transformer Layer
        self.tran_layers = torch.nn.ModuleList()
        for k in range(1, num_layers_tran + 1):
            tran_layer = TranLayer(emb_dim, emb_dim, dropout_rate, attention_dropout_rate, head_size)
            self.tran_layers.append(tran_layer)
            
            
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
#         self.mlp_sign = torch.nn.Linear(emb_dim, num_tasks)
        

    def psi(self, data, config, perturb=None):
#         print('Inside Model:  num graphs: {}, device: {}'.format(batched_data.num_graphs, batched_data.batch.device))

        spin_config = config.to(torch.float32).view(-1,1)
        data.x[:, 0:] = spin_config
    
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[row].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x) + perturb if perturb is not None else self.node_mlp(x)
        h = self.node_mlp(x)
#         x[torch.where(x==-1)] = 0
#         h = self.SpinEncoder(x.long())

        h = self.first_gnn_conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = self.tran_layers[i](h, batch)
            h = gnn_layer(h, edge_index, edge_attr)
            h_list.append(h)
        
        
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        ## two way
#         amplitude = torch.sigmoid(output)
        
#         sign_res = self.mlp_sign(h_graph)
        
#         output = amplitude * torch.cos(sign_res)
       

        return output
    
class GT(torch.nn.Module):
    def __init__(self, num_nodes, num_layers_gnn, num_layers_tran, emb_dim, head_size, dropout_rate = 0, attention_dropout_rate = 0, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):

        super(GT, self).__init__()
        
        self.num_visible = num_nodes
        
        self.JK = JK
        
#         self.node_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        self.SpinEncoder = SpinEncoder(emb_dim)
        
        self.edge_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        

        ### List of GNNs
        self.first_gnn_conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, num_layers_gnn + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
#             act = torch.nn.GELU()
#             act = Swish()


            gnn_layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.gnn_layers.append(gnn_layer)

           
        ### List of Transformer Layer
        self.tran_layers = torch.nn.ModuleList()
        for k in range(1, num_layers_tran + 1):
            tran_layer = TranLayer(emb_dim, emb_dim, dropout_rate, attention_dropout_rate, head_size)
            self.tran_layers.append(tran_layer)
            
            
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
#         self.mlp_sign = torch.nn.Linear(emb_dim, num_tasks)
        

    def psi(self, data, config, perturb=None):
#         print('Inside Model:  num graphs: {}, device: {}'.format(batched_data.num_graphs, batched_data.batch.device))

        data.x = config.to(torch.float32).view(-1,1)
    
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x) + perturb if perturb is not None else self.node_mlp(x)
#         h = self.node_mlp(x)
        x[torch.where(x==-1)] = 0
        h = self.SpinEncoder(x.long())

        h = self.first_gnn_conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = self.tran_layers[i](h, batch)
            h = gnn_layer(h, edge_index, edge_attr)
            h_list.append(h)
        
        
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        ## two way
#         amplitude = torch.sigmoid(output)
        
#         sign_res = self.mlp_sign(h_graph)
        
#         output = amplitude * torch.cos(sign_res)
       

        return output
    
class GT_twoway(torch.nn.Module):
    def __init__(self, num_nodes, num_layers_gnn, num_layers_tran, emb_dim, head_size, dropout_rate = 0, attention_dropout_rate = 0, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):

        super(GT_twoway, self).__init__()
        
        self.num_visible = num_nodes
        
        self.JK = JK
        
        self.node_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        

        ### List of GNNs
        self.first_gnn_conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, num_layers_gnn + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
#             act = torch.nn.GELU()
#             act = Swish()


            gnn_layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.gnn_layers.append(gnn_layer)

           
        ### List of Transformer Layer
        self.tran_layers = torch.nn.ModuleList()
        for k in range(1, num_layers_tran + 1):
            tran_layer = TranLayer(emb_dim, emb_dim, dropout_rate, attention_dropout_rate, head_size)
            self.tran_layers.append(tran_layer)
            
            
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
#         self.mlp_sign = torch.nn.Linear(emb_dim, num_tasks)
        self.sign_mlp = nn.Linear(self.num_visible, 1)
        self.final_norm = torch.nn.LayerNorm(emb_dim)


    def psi(self, data, config, perturb=None):
#         print('Inside Model:  num graphs: {}, device: {}'.format(batched_data.num_graphs, batched_data.batch.device))

        data.x = config.to(torch.float32).view(-1,1)
    
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x) + perturb if perturb is not None else self.node_mlp(x)
        h = self.node_mlp(x)

        h = self.first_gnn_conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = self.tran_layers[i](h, batch)
            h = gnn_layer(h, edge_index, edge_attr)
            h_list.append(h)
        
        
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
#         output = self.graph_pred_linear(h_graph)
        
        ## two way
        output = self.final_norm(h_graph)
        output = self.graph_pred_linear(output)
        amplitude = torch.sigmoid(output)
        
#         sign_res = self.mlp_sign(h_graph)
        
#         output = amplitude * torch.cos(sign_res)
        sign = (self.sign_mlp(config.to(torch.float32))).cos()
        output = amplitude * sign

        return output
    
    
class GT_anti_Sym(torch.nn.Module):
    def __init__(self, num_nodes, num_layers_gnn, num_layers_tran, emb_dim, head_size, dropout_rate = 0, attention_dropout_rate = 0, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):

        super(GT_anti_Sym, self).__init__()
        
        self.num_visible = num_nodes
        
        self.JK = JK
        
        self.node_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, attention_dropout_rate, F.relu)
        

        ### List of GNNs
        self.first_gnn_conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(1, num_layers_gnn + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
#             act = torch.nn.GELU()
#             act = Swish()


            gnn_layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.gnn_layers.append(gnn_layer)

           
        ### List of Transformer Layer
        self.tran_layers = torch.nn.ModuleList()
        for k in range(1, num_layers_tran + 1):
            tran_layer = TranLayer(emb_dim, emb_dim, dropout_rate, attention_dropout_rate, head_size)
            self.tran_layers.append(tran_layer)
            
            
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
#         self.mlp_sign = torch.nn.Linear(emb_dim, num_tasks)
        

    def psi(self, config_in, perturb=None):
#         print('Inside Model:  num graphs: {}, device: {}'.format(batched_data.num_graphs, batched_data.batch.device))

        if config_in[0] == -1:
            config = -config_in
        else:
            config = config_in
    
        data.x = config.to(torch.float32).view(-1,1)
    
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        
        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x) + perturb if perturb is not None else self.node_mlp(x)
        h = self.node_mlp(x)

        h = self.first_gnn_conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = self.tran_layers[i](h, batch)
            h = gnn_layer(h, edge_index, edge_attr)
            h_list.append(h)
        
        
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        ## two way
#         amplitude = torch.sigmoid(output)
        
#         sign_res = self.mlp_sign(h_graph)
        
#         output = amplitude * torch.cos(sign_res)
       
        if config_in[0] == -1:
            return -output
        else:
            return output
        
class CNN1D_sym(nn.Module):
    def __init__(self, num_visible, num_hidden, filter_num=4, kernel_size=3):
        super(CNN1D_sym, self).__init__()
        self.filter_num = filter_num
        self.conv_padding = 1
        self.conv_stride = 1
        self.pooling_stride = 2
        self.conv = nn.Conv1d(1, self.filter_num, 3, stride=1)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        
        out_size = (num_visible + 2 * self.conv_padding - kernel_size) / self.conv_stride + 1
        out_size = (out_size - 2) / self.pooling_stride + 1
        
        self.deconv_size = int(num_visible / out_size)
        
        self.d = nn.Parameter(torch.randn(self.deconv_size,self.filter_num) * 1e-1)
        torch.nn.init.xavier_uniform_(self.d.data)
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        
    def psi(self, data, v_in):
        if v_in[0] == -1:
            v = -v_in
        else:
            v = v_in
        v = v.float().view(1,1,-1)
        v = F.pad(v, (1,1), mode='circular')
        v = self.conv(v)
        v = self.maxpool(v)      
        v = v.squeeze(0).T
        
        c = []
        for i in range(self.d.shape[1]):
            tmp = torch.kron(v[:,i,None],self.d[:,i,None])
            c.append(tmp)

        res = torch.cat(c,axis=1)
        res = torch.sum(res, axis=1).prod()
        
        return res

class GINEVirtual_sublattice(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual_sublattice, self).__init__()
        
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
        
        self.node_mlp = MLP(3, hidden, hidden, dropout, F.relu)
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

    def psi(self, data, config):
        
        spin_config = config.to(torch.float32).view(-1,1)
        data.x[:, 0:] = spin_config
        
        
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
    
    
class GINEVirtual(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual, self).__init__()
        
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

    def psi(self, data, config):
        
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

class GINEVirtual_sym(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual_sym, self).__init__()
        
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

    def psi(self, data, config_in):
        
        if config_in[0] == -1:
            config = -config_in
        else:
            config = config_in
        
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

class SpinEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(SpinEncoder, self).__init__()
        
        self.spin_embedding_list = torch.nn.ModuleList()
        
    
        
        for i, dim in enumerate([2]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.spin_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.spin_embedding_list[i](x[:,i])

        return x_embedding

class DeeperGCN_Virtualnode_sublattice(torch.nn.Module):
    def __init__(self, num_nodes, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperGCN_Virtualnode_sublattice, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling

        self.num_visible = num_nodes
        self.node_mlp = MLP(3, emb_dim, emb_dim, self.drop_ratio, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, self.drop_ratio, F.relu)
        
#         self.SpinEncoder = SpinEncoder(emb_dim)

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()


        ###List of GNNs
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)


            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.layers.append(layer)
            
        for layer in range(num_layers - 1):
            if norm=="batch":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            elif norm=="layer":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU()))
            
            
            
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
        
#         self.sign_mlp = nn.Linear(self.num_visible, 1)
        

    def psi(self, data, config):
        
        spin_config = config.to(torch.float32).view(-1,1)
        data.x[:, 0:] = spin_config
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[row].view(1,-1).to(torch.float32)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

        h = self.node_mlp(x)
#         x[torch.where(x==-1)] = 0
#         h = self.SpinEncoder(x.long())
        
         
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        
        h = h + virtualnode_embedding[batch]
        h = self.layers[0].conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, layer in enumerate(self.layers[1:]):
            h = layer(h, edge_index, edge_attr)
            
            ### update the virtual nodes
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding

            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                    
            h = h + virtualnode_embedding[batch]
            h_list.append(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0, training=self.training)
        
        h_list.append(h)
        h = h + virtualnode_embedding[batch]
        
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
#         sign = (self.sign_mlp(config.float())).cos()
        
        return output
    
    
class DeeperGCN_Virtualnode_twoway(torch.nn.Module):
    def __init__(self, num_nodes, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperGCN_Virtualnode_twoway, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling

        self.num_visible = num_nodes
#         self.node_mlp = MLP(1, emb_dim, emb_dim, self.drop_ratio, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, self.drop_ratio, F.relu)
        
        self.SpinEncoder = SpinEncoder(emb_dim)

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()


        ###List of GNNs
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)


            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.layers.append(layer)
            
        for layer in range(num_layers - 1):
            if norm=="batch":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            elif norm=="layer":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU()))
            
            
            
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
        
        self.sign_mlp = nn.Linear(self.num_visible, 1)
        

    def psi(self, data, config):
        
        data.x = config.to(torch.float32).view(-1,1)
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

#         h = self.node_mlp(x)
        x[torch.where(x==-1)] = 0
        h = self.SpinEncoder(x.long())
        
         
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        
        h = h + virtualnode_embedding[batch]
        h = self.layers[0].conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, layer in enumerate(self.layers[1:]):
            h = layer(h, edge_index, edge_attr)
            
            ### update the virtual nodes
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding

            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                    
            h = h + virtualnode_embedding[batch]
            h_list.append(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0, training=self.training)
        
        h_list.append(h)
        h = h + virtualnode_embedding[batch]
        
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        sign = (self.sign_mlp(config.float())).cos()
        
        return output * sign

class DeeperGCN_Virtualnode_Sym(torch.nn.Module):
    def __init__(self, num_nodes, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", graph_pooling = 'sum', aggr = 'softmax', norm = 'batch', num_tasks=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(DeeperGCN_Virtualnode_Sym, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.graph_pooling = graph_pooling

        self.num_visible = num_nodes
        self.node_mlp = MLP(1, emb_dim, emb_dim, self.drop_ratio, F.relu)
        self.edge_mlp = MLP(1, emb_dim, emb_dim, self.drop_ratio, F.relu)

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()


        ###List of GNNs
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr, t=1.0, learn_t=True, learn_p=False, num_layers=2, norm=norm)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(emb_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(emb_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)


            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0)
            self.layers.append(layer)
            
        for layer in range(num_layers - 1):
            if norm=="batch":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
            elif norm=="layer":
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim, elementwise_affine=True), torch.nn.ReLU()))
            
            
            
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")
            
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            
        
        

    def psi(self, data, config_in):
        
        if config_in[0] == -1:
            config = -config_in
        else:
            config = config_in
        
        data.x = config.to(torch.float32).view(-1,1)
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        row, col = edge_index
        deg = degree(col, data['x'].size(0), dtype=data['x'].dtype)
#         print(deg.shape)
#         print(row.shape)
#         print(deg[row].shape)
#         edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0).to(device)
        edge_attr = deg[col].view(1,-1)#.to(device)
#         print(edge_attr.shape)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        
        ### computing input node embedding
        edge_attr = self.edge_mlp(edge_attr)

        h_list = []

        h = self.node_mlp(x)
        
         
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        
        h = h + virtualnode_embedding[batch]
        h = self.layers[0].conv(h, edge_index, edge_attr)
        
        h_list.append(h)
        for i, layer in enumerate(self.layers[1:]):
            h = layer(h, edge_index, edge_attr)
            
            ### update the virtual nodes
            ### add message from graph nodes to virtual nodes
            virtualnode_embedding_temp = global_add_pool(h, batch) + virtualnode_embedding

            ### transform virtual nodes using MLP
            virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                    
            h = h + virtualnode_embedding[batch]
            h_list.append(h)

        h = self.layers[0].act(self.layers[0].norm(h))
        h = F.dropout(h, p=0, training=self.training)
        
        h_list.append(h)
        h = h + virtualnode_embedding[batch]
        
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]
                
        h_graph = self.pool(node_representation, data.batch)
        output = self.graph_pred_linear(h_graph)
        
        return output

class GINEVirtual_twoway_sublattice(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual_twoway_sublattice, self).__init__()
        
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
#         self.mlp_sign = MLP(hidden, hidden, 1, dropout, F.relu)
        
        self.node_mlp = MLP(3, hidden, hidden, dropout, F.relu)
        self.edge_mlp = MLP(2, hidden, hidden, dropout, F.relu)
#         self.appnp = APPNP(3,0.1)
        
#         self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(hidden, 2*hidden), torch.nn.BatchNorm1d(2*hidden), torch.nn.ReLU(), torch.nn.Linear(2*hidden, 1)))
        
       
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        
        
        for layer in range(self.depth):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden), torch.nn.ReLU(), \
                                                    torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden), torch.nn.ReLU()))
            
        self.sign_mlp = nn.Linear(self.num_visible, 1)


    def psi(self, data, config):
        
        spin_config = config.to(torch.float32).view(-1,1)
        data.x[:, 0:] = spin_config
        
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
        x = global_add_pool(x, batch)
#         x = self.pool(x, batch)
#         z[-1] = x
#         x = torch.sum(torch.cat(z, 0), 0)    
        
        # regression
        #x = F.dropout(x, p=self.dropout, training=self.training)
        amplitude = self.mlp(x)
        amplitude = torch.sigmoid(amplitude)
        
#         sign_res = self.mlp_sign(x)
        sign = (self.sign_mlp(config.float())).cos()
        x = amplitude * sign
#         if self.training:
#             return output
#         else:
#             # At inference time, relu is applied to output to ensure positivity
#             return torch.clamp(output, min=float('-inf'), max=0)
        return x

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
    
class GINEVirtual_twoway(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual_twoway, self).__init__()
        
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
        self.mlp_sign = MLP(hidden, hidden, 1, dropout, F.relu)
        
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

    def psi(self, data, config):
        
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
        amplitude = self.mlp(x)
        amplitude = torch.sigmoid(amplitude)
        
        sign_res = self.mlp_sign(x)
        
        x = amplitude * torch.cos(sign_res)
#         if self.training:
#             return output
#         else:
#             # At inference time, relu is applied to output to ensure positivity
#             return torch.clamp(output, min=float('-inf'), max=0)
        return x

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
    
class GINEVirtual_twoway_sym(torch.nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, hidden = 32, dropout = 0, depth = 3, num_spins = 10):
        super(GINEVirtual_twoway_sym, self).__init__()
        
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
        self.mlp_sign = MLP(hidden, hidden, 1, dropout, F.relu)
        
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

    def psi(self, data, config_in):

        if config_in[0] == -1:
            config = -config_in
        else:
            config = config_in
        
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
        amplitude = self.mlp(x)
        amplitude = torch.sigmoid(amplitude)
        
        sign_res = self.mlp_sign(x)
        
        x = amplitude * torch.cos(sign_res)
#         if self.training:
#             return output
#         else:
#             # At inference time, relu is applied to output to ensure positivity
#             return torch.clamp(output, min=float('-inf'), max=0)
        return x

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

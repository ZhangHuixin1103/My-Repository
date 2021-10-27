import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy import sparse
from numpy.random import randn
import os
import random
import torch
from torch_scatter import scatter_sum
from torch_geometric.utils import from_networkx, degree, sort_edge_index

def get_undirected_idx_list(data):
    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0,i] < data.edge_index[1,i]:
            idx_list.append(data.edge_index[:,i].numpy().tolist())
    return idx_list
        
def flip(s, idx):
    sflip = deepcopy(s)
    sflip[idx] = -sflip[idx]
    return sflip

def get_dist2_idx(idx_list):
    idx_list_dist2 = []
    for idx in idx_list:
        for idx2 in idx_list:
            if idx[0] == idx2[0]:
                a, b = idx[1], idx2[1]
            elif idx[1] == idx2[0]:
                a, b = idx[0], idx2[1]
            elif idx[0] == idx2[1]:
                a, b = idx[1], idx2[0]
            elif idx[1] == idx2[1]:
                a, b = idx[0], idx2[0]
            else:
                a, b = None, None
            if a is not None and a != b:
                if b < a:
                    a, b = b, a
                if [a, b] not in idx_list and [a, b] not in idx_list_dist2:
                    idx_list_dist2.append([a, b])
    return idx_list_dist2

def heisenberg_loc(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
    
    # sigma_z * sigma_z
    e_part1 = 0
    for idx in idx_list:
        if config[idx[0]] == config[idx[1]]:
            e_part1 += 1
        else:
            e_part1 -= 1

  
    # sigma_x * sigma_x
    e_part2 = 0
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        e_part2 += (psi_func(config_i) / psi_loc)
        
        
    # sigma_y * sigma_y
    e_part3 = 0
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        if config[idx[0]] == config[idx[1]]:
            e_part3 -= (psi_func(config_i) / psi_loc)
        else:
            e_part3 += (psi_func(config_i) / psi_loc)

    idx_list_dist2 = get_dist2_idx(idx_list)

    # sigma_z * sigma_z
    e_part1_dist2 = 0
    for idx in idx_list_dist2:
        if config[idx[0]] == config[idx[1]]:
            e_part1_dist2 += 1
        else:
            e_part1_dist2 -= 1

    # sigma_x * sigma_x
    e_part2_dist2 = 0
    for idx in idx_list_dist2:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        e_part2_dist2 += (psi_func(config_i) / psi_loc)

    # sigma_y * sigma_y
    e_part3_dist2 = 0
    for idx in idx_list_dist2:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        if config[idx[0]] == config[idx[1]]:
            e_part3_dist2 -= (psi_func(config_i) / psi_loc)
        else:
            e_part3_dist2 += (psi_func(config_i) / psi_loc)

    return e_part1 + e_part2 + e_part3 + J * (e_part3_dist2 + e_part2_dist2 + e_part1_dist2)

def heisenberg_loc_fast(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):

    # sigma_z * sigma_z
    idx_array = np.array(idx_list)
    mask = config[idx_array[:, 0]] != config[idx_array[:, 1]]
    e_part1 = len(idx_array) - sum(mask)

    new_config_list = []
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        new_config_list.append(config_i)

    # calculate wave function value in new config list
    psi_list = psi_func(new_config_list)
    psi_ratio_list = psi_list / psi_loc

    # sigma_x * sigma_x
    e_part2 = sum(psi_ratio_list)

    mask = np.array(list(map(int, mask))) * 2 - 1

    # sigma_y * sigma_y
    e_part3 = sum(psi_ratio_list * mask)

    return e_part1 + e_part2 + e_part3

def tensor_prod_graph_Heisenberg(data, J=-1, h=-0.5, n=10):
    def tensor_prod(idx, s, size=10, J=-1, h=-0.5):
        "Tensor product of `s` acting on indexes `idx`. Fills rest with Id."
        Id = np.array([[1, 0], [0, 1]])
        idx, s = np.array(idx), np.array(s)
        matrices = [Id if k not in idx else s for k in range(size)]
        prod = matrices[0]
        for k in range(1, size):
            prod = np.kron(prod, matrices[k])
        return prod

    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    sy = np.array([[0, -1j], [1j, 0]])

    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            idx_list.append(edge_index[:, i].numpy().tolist())

    H_1 = sum([tensor_prod(idx, sz, size=n) for idx in idx_list])
    H_2 = sum([tensor_prod(idx, sx, size=n) for idx in idx_list])
    H_3 = sum([tensor_prod(idx, sy, size=n) for idx in idx_list])

    idx_list_dist2 = get_dist2_idx(idx_list)
    H_1_dist2 = sum([tensor_prod(idx, sz, size=n) for idx in idx_list_dist2])
    H_2_dist2 = sum([tensor_prod(idx, sx, size=n) for idx in idx_list_dist2])
    H_3_dist2 = sum([tensor_prod(idx, sy, size=n) for idx in idx_list_dist2])

    H = (H_1 + H_2 + H_3) + J * (H_1_dist2 + H_2_dist2 + H_3_dist2)

    return H


def tensor_prod_graph_Heisenberg_sparse(data, J=-1, h=-0.5, n=10):
    def tensor_prod(idx, s, size=10, J=-1, h=-0.5):
        "Tensor product of `s` acting on indexes `idx`. Fills rest with Id."
        Id = identity(2)
        idx = np.array(idx)
        matrices = [Id if k not in idx else s for k in range(size)]
        prod = matrices[0]
        for k in range(1, size):
            prod = sparse.kron(prod, matrices[k])
        return prod

    sx = csr_matrix(np.array([[0, 1], [1, 0]]))
    sz = csr_matrix(np.array([[1, 0], [0, -1]]))
    sy = csr_matrix(np.array([[0, -1j], [1j, 0]]))

    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            idx_list.append(edge_index[:, i].numpy().tolist())

    H_1 = csr_matrix(np.zeros((2**n,2**n)))
    for idx in idx_list:
        H_1 = H_1 + tensor_prod(idx, sz, size=n)

    H_2 = csr_matrix(np.zeros((2**n,2**n)))
    for idx in idx_list:
        H_2 = H_2 + tensor_prod(idx, sx, size=n)

    H_3 = csr_matrix(np.zeros((2**n,2**n)))
    for idx in idx_list:
        H_3 = H_3 + tensor_prod(idx, sy, size=n)

    H = (H_1 + H_2 + H_3)

    return H

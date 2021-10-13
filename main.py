import numpy as np
import os
import networkx as nx
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

from torch_geometric.utils import from_networkx, degree, sort_edge_index
from torch_geometric.nn import GATConv, GraphConv, GCNConv, GINConv, GINEConv, Set2Set, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, LayerNorm, BatchNorm, GlobalAttention

from model import CNN2D

### Adaptively adjust from https://github.com/GiggleLiu/marburg

num_spin = 100
num_hidden = 10
epsilon = 1e-15

data = None


def tensor_prod_graph_Heisenberg(data, J=-1, h=-0.5, n=10):
    def tensor_prod(idx, s, size=10, J=-1, h=-0.5):

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
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0, i] < data.edge_index[1, i]:
            idx_list.append(data.edge_index[:, i].numpy().tolist())

    H_1 = sum([tensor_prod(idx, sz, size=n) for idx in idx_list])
    H_2 = sum([tensor_prod(idx, sx, size=n) for idx in idx_list])
    H_3 = sum([tensor_prod(idx, sy, size=n) for idx in idx_list])
    H = (H_1 + H_2 + H_3)
    return H


def vmc_sample(kernel, initial_config, num_bath, num_sample):
    '''
    obtain a set of samples.

    Args:
        kernel (object): defines how to sample, requiring the following methods:
            * propose_config, propose a new configuration.
            * prob, get the probability of specific distribution.
        initial_config (1darray): initial configuration.
        num_bath (int): number of updates to thermalize.
        num_sample (int): number of samples.

    Return:
        list: a list of spin configurations.
    '''
    print_step = np.Inf  # steps between two print of accept rate, Inf to disable showing this information.

    config = initial_config
    log_prob = kernel.log_prob(config)

    n_accepted = 0
    sample_list = []
    for i in range(num_bath + num_sample):
        #         print('sample ', i)
        # generate new config and calculate probability ratio
        config_proposed = kernel.propose_config(config)
        log_prob_proposed = kernel.log_prob(config_proposed)

        # accept/reject a move by metropolis algorithm
        if np.random.random() < np.exp(log_prob_proposed - log_prob):
            config = config_proposed
            log_prob = log_prob_proposed
            n_accepted += 1

        # print statistics
        if i % print_step == print_step - 1:
            print('%-10s Accept rate: %.3f' %
                  (i + 1, n_accepted * 1. / print_step))
            n_accepted = 0

        # add a sample
        if i >= num_bath:
            sample_list.append(config)
    return sample_list


def vmc_measure(local_measure, sample_list, measure_step, num_bin=50):
    '''
    perform measurements on samples

    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.
        meaure_step: number of samples skiped between two measurements + 1.

    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    energy_loc_list, grad_loc_list = [], []
    for i, config in enumerate(sample_list):
        if i % measure_step == 0:
            # back-propagation is used to get gradients.
            energy_loc, grad_loc = local_measure(config)
            energy_loc_list.append(energy_loc)
            grad_loc_list.append(grad_loc)
            # print(grad_loc, energy_loc)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)
    # print(energy_loc_list, energy)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        #print(grad_loc)
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))
    return energy.item(), grad_mean, energy_grad, energy_precision


def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    num_sample = len(var_list)
    if num_sample % num_bin != 0:
        raise
    size_bin = num_sample // num_bin

    # mean, variance
    mean = np.mean(var_list, axis=0)
    variance = np.var(var_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var(
        [np.mean(var_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin * \
             np.abs(np.mean(variance_binned) / np.mean(variance))
    stderr = np.sqrt(variance_binned / num_bin)
    print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
          (mean, stderr, t_auto))
    return mean, stderr


class VMCKernel(object):
    '''
    variational monte carlo kernel.

    Attributes:
        energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
        ansatz (Module): torch neural network.
    '''

    def __init__(self, data, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc
        self.data = data

    def prob(self, config):
        '''
        probability of configuration.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number: probability |<config|psi>|^2.
        '''
        return abs(self.ansatz.psi(self.data, torch.from_numpy(config)).item()) ** 2

    def log_prob(self, config):
        sign, log_prob = self.ansatz.log_psi(self.data, torch.from_numpy(config))
        return log_prob.item() * 2

    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number, list: local energy and local gradients for variables.
        '''
        psi_loc = self.ansatz.psi(self.data, torch.from_numpy(config))

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        psi_loc.backward()
        grad_loc = [p.grad.data / psi_loc.item() for p in self.ansatz.parameters()]
        #grad_loc = [p.grad.data for p in self.ansatz.parameters()]

        # E_{loc}
        edge_index = sort_edge_index(self.data.edge_index)[0]
        idx_list = []
        for i in range(self.data.edge_index.shape[1]):
            if self.data.edge_index[0, i] < self.data.edge_index[1, i]:
                idx_list.append(self.data.edge_index[:, i].numpy().tolist())
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(self.data, torch.from_numpy(x)).data, psi_loc.data,
                               idx_list)
        # print(eloc.item(), psi_loc.item(), grad_loc)
        return eloc.item(), grad_loc

    @staticmethod
    def propose_config(old_config):
        '''
        flip two positions as suggested spin flips.

        Args:
            old_config (1darray): spin configuration, which is a [-1,1] string.

        Returns:
            1darray: new spin configuration.
        '''

        num_spin = len(old_config)
        upmask = old_config == 1
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = -1
        config[iflip1] = 1

        # randomly flip one of the spins
        #         def flip(config, idx):
        #             new_config = config.copy()
        #             new_config[idx] = -new_config[idx]
        #             return new_config

        #         idx = np.random.randint(0, len(old_config), 1)
        #         config = flip(old_config, idx)

        return config


def flip(s, idx):
    sflip = deepcopy(s)
    sflip[idx] = -sflip[idx]
    return sflip


def heisenberg_loc(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
    # print(psi_loc)

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
        #if abs(e_part2) > 1e15:
        #    e_part2 = np.sign(e_part2) * 1e15

    # sigma_y * sigma_y
    e_part3 = 0
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        if config[idx[0]] == config[idx[1]]:
            e_part3 -= (psi_func(config_i) / psi_loc)
        else:
            e_part3 += (psi_func(config_i) / psi_loc)
        #if abs(e_part3) > 1e16:
        #    e_part3 = np.sign(e_part3) * 1e16

    return e_part1 + e_part2 + e_part3


# def get_wave_function(model, config_list):
#     psi_vec = []
#     for config in config_list:
#         config = np.array(config)
#         psi = model.ansatz.psi(torch.from_numpy(config)).detach().numpy()
#         psi_vec.append(psi)
#     return np.array(psi_vec)
# l = []
# config_list = []
# def gen(l, num_spin, config_list):
#     if num_spin == 0:
#         config_list.append(l)
#         return
#     gen(l + [1], num_spin-1, config_list)
#     gen(l + [-1], num_spin-1, config_list)
# gen(l, num_spin, config_list) 


def train(model, num_spin):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.

    '''

    initial_config = np.array([-1, 1] * (model.ansatz.num_visible // 2))

    step = 0
    lr = 0.001

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.        
        sample_list = vmc_sample(model, initial_config, num_bath=100 * num_spin, num_sample=100 * num_spin)
        energy, grad, energy_grad, precision = vmc_measure(model.local_measure, sample_list, num_spin)
        # print(energy, grad, energy_grad, precision)

        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]

        # update parameter using SGD
        #         for var, g in zip(model.ansatz.parameters(), g_list):
        #             delta = lr * g
        #             var.data -= delta

        # update parameter using adam
        t = 1
        eps = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        sqrs = []
        vs = []
        for param in model.ansatz.parameters():
            sqrs.append(torch.zeros_like(param.data))
            vs.append(torch.zeros_like(param.data))

        print('learning rate: ', lr)
        for param, g, v, sqr in zip(model.ansatz.parameters(), g_list, vs, sqrs):
            v[:] = beta1 * v + (1 - beta1) * g
            sqr[:] = beta2 * sqr + (1 - beta2) * g ** 2
            v_hat = v / (1 - beta1 ** t)
            s_hat = sqr / (1 - beta2 ** t)
            param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)

        step += 1
        if step % 20 == 0:
            if lr > 0.0001:
                print('lr decay!!!')
                lr *= 0.5
            else:
                lr = 0.0001

        yield energy, precision


def main():
    parser = argparse.ArgumentParser(description='quantum many body problem with variational monte carlo method')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='cnn2d',
                        help='RBM, gine-res-virtual, cnn1d, cnn2d, or gnn-transformer (default: gine-res-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum or attention (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 32)')
    #     parser.add_argument('--batch_size', type=int, default=256,
    #                         help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="../log/cnn2d",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default='../data/10_10_2d_lattice.pt',
                        help='directory to load graph data')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_dir', type=str, default='', help='directory to save model parameter and energy list')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    ### load data and get undirected unique edge index list
    global data
    data = torch.load(args.data_dir)
    print(data)

    ### get hamiltonian matrix and ground truth energy
    if num_spin < 16:
        H = tensor_prod_graph_Heisenberg(data, n=num_spin)
        e_vals, e_vecs = np.linalg.eigh(H)
        E_exact = e_vals[0]

    ### visualize the loss history
    energy_list, precision_list = [], []

    def _update_curve(energy, precision):
        energy_list.append(energy)
        precision_list.append(precision)
        # if len(energy_list)%10 == 0:
        #     plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
        #     # dashed line for exact energy
        #     plt.axhline(E_exact, ls='--')
        #     plt.show()

    params = {
        'num_spins': 10,
        'hidden': 32,
        'dropout': 0,
        'depth': 3
    }

    # rbm = RBM(num_spin, num_hidden)
    # gnn = GNNvirtual(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
    if args.model == 'gnn-transformer':
        pass
        # row, col = data.edge_index
        # deg = degree(col, data['x'].size(0))
        # edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0)
        # data.edge_attr = torch.transpose(edge_attr, 0, 1)
        # ansatz = GraphTransformer(data, max(deg).item(), data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32,
        #                    head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
    elif args.model == 'cnn2d':
        ansatz = CNN2D(num_spin, num_hidden, filter_num=32, kernel_size=5).cuda()
    model = VMCKernel(data, heisenberg_loc, ansatz=ansatz)

    ### load pretrain model 
    # path = '../result/Heisenberg/chain_4_node/GNNTrans/model.pt'
    # model.ansatz.load_state_dict(torch.load(path))

    num_params = sum(p.numel() for p in model.ansatz.parameters())
    print(f'#Params: {num_params}')

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    t0 = time.time()
    for i, (energy, precision) in enumerate(train(model, num_spin)):
        t1 = time.time()
        print('Step %d, Energy = %.4f, elapse = %.4f' % (i, energy, t1 - t0))
        _update_curve(energy, precision)
        t0 = time.time()

        if args.log_dir != '':
            writer.add_scalar('energy', energy, i)
            writer.add_scalar('precision', precision, i)

        if args.save_dir != '':
            #             save_dir = '../result/Heisenberg/chain_8_node/GNNTrans/'
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            energy_save_dir = os.path.join(save_dir, 'graph_energy_list.npy')
            precision_save_dir = os.path.join(save_dir, 'graph_precision_list.npy')
            np.save(energy_save_dir, energy_list)
            np.save(precision_save_dir, precision_list)
            torch.save(model.ansatz.state_dict(), save_dir + 'model.pt')

        # stop condition
        if i >= args.epochs:
            break


if __name__ == "__main__":
    main()

import sys
sys.path.append("..")

import numpy as np
import os
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
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
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool, LayerNorm, BatchNorm, GlobalAttention

# from methods.models import GT, CNN1D, GINEVirtual, RBM, CNN1D_sym, GINEVirtual_sym, DeeperGCN_Virtualnode_Sym, GINEVirtual_twoway, GINEVirtual_twoway_sym, RBM_twoway, Model_twoway_invariant, RBM_mul_twoway
from methods.models import *
from methods.vmc import vmc_sample, vmc_measure, VMCKernel, vmc_measure_two_path
from utils.utils import tensor_prod_graph_Heisenberg, heisenberg_loc, get_undirected_idx_list, tensor_prod_graph_Heisenberg_sparse

### Adaptively adjust from https://github.com/GiggleLiu/marburg


num_spin = 10
num_hidden = 10

data = None

def train(model, num_spin, optimizer):
    '''
    train a model.
    Args:
        model (obj): a model that meet VMC model definition.
        num_spin: number of spin sites
        optimizer: pytorch optimizer 
    '''

    initial_config = np.array([-1, 1] * (model.ansatz.num_visible // 2))
    
    # get expectation values for energy, gradient and their product,
    # as well as the precision of energy.    
    start = time.time()
    sample_list = vmc_sample(model, initial_config, num_bath=100*num_spin, num_sample=200*num_spin)
    print("vmc sample time: ", time.time() - start)
    start = time.time()
    energy, grad, energy_grad, precision = vmc_measure(data, model.local_measure, sample_list, 1*num_spin)
    # energy, grad_log, energy_grad, precision = vmc_measure_two_path(data, model.local_measure_two_path, sample_list, 1 * num_spin)
    print("stochastic estimate time: ", time.time() - start)

    g_list = [(eg - energy * g)*2 for eg, g in zip(energy_grad, grad)]
    # g_list = [2 * eg - 2 * energy * g for eg, g in zip(energy_grad, grad_log)]

    # update parameter using SGD
#         for var, g in zip(model.ansatz.parameters(), g_list):
#             delta = lr * g
#             var.data -= delta

    optimizer.zero_grad()
    for param, g in zip(model.ansatz.parameters(), g_list):
        param.grad.data = g.to(torch.float)
    optimizer.step()    

    # update parameter using adam
#         t = 1
#         eps = 1e-8
#         beta1=0.9
#         beta2=0.999
#         sqrs = []
#         vs = []
#         for param in model.ansatz.parameters():
#             sqrs.append(torch.zeros_like(param.data))
#             vs.append(torch.zeros_like(param.data))

#         print('learning rate: ', lr)
#         for param, g, v, sqr in zip(model.ansatz.parameters(), g_list, vs, sqrs):
#             v[:] = beta1 * v + (1 - beta1) * g
#             sqr[:] = beta2 * sqr + (1 - beta2) * g ** 2
#             v_hat = v / (1 - beta1 ** t)
#             s_hat = sqr / (1 - beta2 ** t)
#             param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)

    return energy, precision
        
        
def main():

    parser = argparse.ArgumentParser(description='quantum many body problem with variational monte carlo method')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='cnn',
                        help='RBM, gine-res-virtual, cnn1d, cnn2d, or gnn-transformer (default: gine-res-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum or attention (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 32)')
    parser.add_argument('--num_spin', type=int, default=8,
                            help='spin number')
#     parser.add_argument('--batch_size', type=int, default=256,
#                         help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="../log/gnn-transformer",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default = '../dataset/10_node_chain.pt', help='directory to load graph data')
    parser.add_argument('--dataname', type=str, default = '10_node_chain', help='data name')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_dir', type=str, default = '../result/Heisenberg/chain_8_node/GNNTrans/', help='directory to save model parameter and energy list')
    parser.add_argument('--invariant', action='store_true', default=False, help="whether use invariant")

    parser.add_argument('--J', type=float, default=0.0, help='J1/J2')

    args = parser.parse_args()

    print(args)

    seed = 42 #10086
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    ### load data and get undirected unique edge index list
    global data
    datapath = os.path.join(args.data_dir, args.dataname + '.pt')
#     print(datapath)
    data = torch.load(datapath)

    global num_spin, num_hidden
    num_spin = args.num_spin
    num_hidden = args.num_spin
    
    ### get hamiltonian matrix and ground truth energy
    E_exact = None
    if num_spin <= 16:
        H = tensor_prod_graph_Heisenberg(data, n=num_spin, J=args.J)
        # e_vals, e_vecs = np.linalg.eigh(H)
        [energy,state]=eigsh(H,k=1,which='SA')
        E_exact = energy #e_vals[0]
        print('Exact energy: {}'.format(E_exact))
    
        
    ### visualize the loss history
    energy_list, precision_list = [], []
    def _update_curve(energy, precision, save_dir):
        energy_list.append(energy)
        precision_list.append(precision)
        if len(energy_list)%10 == 0:
            fig = plt.figure()
            plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
            # dashed line for exact energy
            plt.axhline(E_exact, ls='--')
        #     plt.show()
            fig.savefig(save_dir + 'energy.png')

    params = {
        'num_spins': args.num_spin,
        'hidden': args.emb_dim,
        'dropout': args.drop_ratio,
        'depth': args.num_layers
    }


    if args.invariant == True:
        ansatz = Model_twoway_invariant(data, num_spin, num_hidden, args.model)
    else:
        if args.model == 'graph-transformer':
            row, col = data.edge_index
            deg = degree(col, data['x'].size(0))
            edge_attr = torch.cat((deg[row].view(1,-1),deg[col].view(1,-1)), dim=0)
            data.edge_attr = torch.transpose(edge_attr, 0, 1)
            ansatz = GraphTransformer(data, max(deg).item(), data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32,
                               head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif args.model == 'gnn-transformer':
            ansatz = GT(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif args.model == 'gnn-transformer-sublattice':
            ansatz = GT_sublattice(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif args.model == 'gnn-transformer-twoway':
            ansatz = GT_twoway(data.num_nodes, num_layers_gnn=4//2, num_layers_tran=4//2, emb_dim=32, 
                       head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum', norm='batch')
        elif args.model == 'cnn2d':
            ansatz = CNN2D(data.num_nodes, num_hidden, filters_in=1, filters=3, kernel_size=3)
        elif args.model == 'cnn1d':
            ansatz = CNN1D(num_spin, num_hidden, filter_num=4, kernel_size=3)
        elif args.model == 'cnn1d-twoway':
            ansatz = CNN1D_twoway(num_spin, num_hidden, filter_num=4, kernel_size=3)
        elif args.model == 'cnn1d-sym':
            ansatz = CNN1D_sym(num_spin, num_hidden, filter_num=4, kernel_size=3)
        elif args.model == 'FFN-twoway':
            ansatz = FFN_twoway(num_spin, num_hidden)
        elif args.model == 'FFN-twoway-sigmoid':
            ansatz = FFN_twoway_sigmoid(num_spin, num_hidden)
        elif args.model == 'RBM':
            ansatz = RBM(num_spin, num_hidden)
        elif args.model == 'RBM-twoway':
            ansatz = RBM_twoway(num_spin, num_hidden)
#         elif args.model == 'RBM-twoway-invariant':
#             ansatz = Model_twoway_invariant(num_spin, num_hidden, args.model)
        elif args.model == 'RBM-mul-twoway':
            ansatz = RBM_mul_twoway(num_spin, num_hidden)
#         elif args.model == 'RBM-mul-twoway-invariant':
#             ansatz = RBM_mul_twoway_invariant(num_spin, num_hidden, args.model)
        elif args.model == 'gine-res-virtual':
            ansatz = GINEVirtual(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-sublattice':
            ansatz = GINEVirtual_sublattice(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-sym':
            ansatz = GINEVirtual_sym(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'deepergcn-virtual-sym':
            ansatz = DeeperGCN_Virtualnode_Sym(data.num_nodes,
                                   num_layers=5, 
                                   emb_dim=32, 
                                   drop_ratio = 0, 
                                   JK = 'last', 
                                   graph_pooling='sum', 
                                   norm='layer')
        elif args.model == 'deepergcn-virtual-twoway':
            ansatz = DeeperGCN_Virtualnode_twoway(data.num_nodes,
                                   num_layers=5, 
                                   emb_dim=32, 
                                   drop_ratio = 0, 
                                   JK = 'last', 
                                   graph_pooling='sum', 
                                   norm='layer')
        elif args.model == 'deepergcn-virtual-sublattice':
            ansatz = DeeperGCN_Virtualnode_sublattice(data.num_nodes,
                                   num_layers=5, 
                                   emb_dim=32, 
                                   drop_ratio = 0, 
                                   JK = 'last', 
                                   graph_pooling='sum', 
                                   norm='layer')
        elif args.model == 'gine-res-virtual-two-way-sym':
            ansatz = GINEVirtual_twoway_sym(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-two-way':
            ansatz = GINEVirtual_twoway(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-two-way-sublattice':
            ansatz = GINEVirtual_twoway_sublattice(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        else:
            raise ValueError('Invalid model type')
        
    model = VMCKernel(data, heisenberg_loc, ansatz=ansatz, J=args.J)
    
    optimizer = torch.optim.Adam(model.ansatz.parameters(), lr=0.01)
    
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    
    
    ### load pretrain model 
    # path = '../result/Heisenberg/chain_4_node/GNNTrans/model.pt'
    # model.ansatz.load_state_dict(torch.load(path))
#     checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataname, args.model, '')
#     checkpoint = torch.load(checkpoint_dir + 'checkpoint.pt')

#     # ansatz = RBM_twoway(num_spin, num_hidden)
#     # ansatz = RBM(num_spin, num_hidden)
#     ansatz = Model_twoway_invariant(data, num_spin, num_hidden, args.model)
#     model = VMCKernel(data, heisenberg_loc, ansatz=ansatz)
#     model.ansatz.load_state_dict(checkpoint['model_state_dict'])
    
    
    num_params = sum(p.numel() for p in model.ansatz.parameters())
    print(f'#Params: {num_params}')
    
    if args.log_dir != '':
        log_dir = os.path.join(args.log_dir, args.dataname, args.model, '')
        writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    for step in range(args.epochs):
        energy, precision = train(model, num_spin, optimizer)
        t1 = time.time()
        # print('Step %d, dE/|E| = %.4f, elapse = %.4f' % (step, -(energy - E_exact)/E_exact, t1-t0))
        print('Step %d, E = %.4f, elapse = %.4f' % (step, energy, t1 - t0))
        t0 = time.time()
        
        if args.log_dir != '':
            if E_exact is None:
                writer.add_scalars('energy', {'pred': energy}, step)
            else:
                writer.add_scalars('energy', {'pred':energy, 'gt':E_exact}, step)
            

        if args.save_dir != '':
#             save_dir = '../result/Heisenberg/chain_8_node/'
            save_dir = os.path.join(args.save_dir, args.dataname, args.model, '')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            _update_curve(energy, precision, save_dir)
            energy_save_dir = os.path.join(save_dir, 'graph_energy_list.npy')
            precision_save_dir = os.path.join(save_dir, 'graph_precision_list.npy')
            np.save(energy_save_dir, energy_list)
            np.save(precision_save_dir, precision_list)
        
        if args.checkpoint_dir != '':
            checkpoint = {'step': step, 'model_state_dict': model.ansatz.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'energy': energy, 'num_params': num_params}

            checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataname, args.model, '')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(checkpoint, checkpoint_dir + 'checkpoint.pt')
        
        if optimizer.param_groups[0]['lr'] > 0.0001:
            scheduler.step()
    

if __name__ == "__main__":
    main()

import sys

sys.path.append("..")

import numpy as np
import os
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

from torch_geometric.utils import degree

from methods.models import *
from methods.vmc_batch_tensor_new_log_arg import vmc_sample_batch, VMCKernel
from utils.utils_log_arg import tensor_prod_graph_Heisenberg, get_undirected_idx_list, heisenberg_loc_batch_fast_J1_J2, \
                        heisenberg_loc_it_swo_J1_J2

### Adaptively adjust from https://github.com/GiggleLiu/marburg

num_spin = 10
num_hidden = 10

data = None

def test(model, num_spin, idx_list, J2, device='cpu'):
    '''
    test model
    Args:
        model: trained model for testing
        num_spin: number of spin sites
        idx_list: unique edge index list
        device: gpu or cpu
    Returns: None
    '''

    total_sample = 200000
    batch_size = 200

    initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
    initial_config_batch = initial_config.unsqueeze(0).to(device).repeat(batch_size, 1) # torch.tile(initial_config, (batch_size, 1)).to(device)

    np.random.seed(100)
    indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
    initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]

    sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath=1000 * num_spin,
                                    num_sample=batch_size)

    energy = 0
    bin_energy = 0
    total_step = int(total_sample / batch_size)
    bin = []
    for step in range(1, total_step+1):
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin * 1, num_sample=batch_size)
        with torch.no_grad():
            # psi_loc = model.ansatz.psi_batch(data, sample_batch)
            # eloc = model.energy_loc(sample_batch, lambda x: model.ansatz.psi_batch(data, x).data, psi_loc.data,
            #                        idx_list, J2)
            log_psi_loc, arg_psi_loc = model.ansatz.psi_batch(data, sample_batch)
            eloc = model.energy_loc(sample_batch, psi_func=lambda x: model.ansatz.psi_batch(data, x),
                                   log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                   idx_list=idx_list, J2=J2)

            energy += (eloc.mean() / num_spin)
            bin_energy += (eloc.mean() / num_spin)
        if step % 20 == 0:
            bin.append(bin_energy / 20)
            bin_energy = 0

    energy = energy / total_step
    std = torch.std(torch.stack(bin))
    print(bin)
    print('Test Energy: ', energy, 'std: ', std)

def train_it_swo(model, num_spin, idx_list, optimizer, J2, first_batch=False, sample_batch=None, ansatz_phi=None, device='cpu'):
    '''
    train a model.
    Args:
        model (obj): a model that meet VMC model definition.
        num_spin: number of spin sites
        optimizer: pytorch optimizer
    '''
    batch_size = 200
    # Sample
    if first_batch:
        initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
        initial_config_batch = initial_config.unsqueeze(0).to(device).repeat(batch_size, 1)
        # initial_config_batch = torch.tile(initial_config, (batch_size, 1)).to(device)

        indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
        initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]
        sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath=1000 * num_spin, num_sample=batch_size)
    else:
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin, num_sample=batch_size)

    # Estimate gradient
    energy, grad = model.local_measure_it_swo(config=sample_batch, idx_list=idx_list, J2=J2, beta=0.05, ansatz_phi=ansatz_phi)

    optimizer.zero_grad()
    for param, g in zip(model.ansatz.parameters(), grad):
        param.grad.data = g.float()
    optimizer.step()

    precision = 0
    return energy.real.cpu(), precision, sample_batch


def train(model, num_spin, idx_list, optimizer, J2, first_batch=False, sample_batch=None, device='cpu'):
    '''
    train a model.
    Args:
        model (obj): a model that meet VMC model definition.
        num_spin: number of spin sites
        optimizer: pytorch optimizer
    '''
    batch_size = 300
    # Sample
    if first_batch:
        initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
        initial_config_batch = initial_config.unsqueeze(0).to(device).repeat(batch_size, 1)
        # initial_config_batch = torch.tile(initial_config, (batch_size, 1)).to(device)

        indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
        initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]
        sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath= 1000 * num_spin, num_sample=batch_size)
    else:
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin, num_sample=batch_size)

    # Estimate gradient
    energy, grad = model.local_measure_two_path(config=sample_batch, idx_list=idx_list, J2=J2)

    optimizer.zero_grad()
    for param, g in zip(model.ansatz.parameters(), grad):
        param.grad.data = g.float()
    optimizer.step()

    precision = 0
    return energy.real.cpu(), precision, sample_batch



def main():
    parser = argparse.ArgumentParser(description='quantum many body problem with variational monte carlo method')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--J2', type=float, default=0.0,
                        help='J2 value in Heisenberg model')
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    #     parser.add_argument('--batch_size', type=int, default=256,
    #                         help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="../log/gnn-transformer",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default='../dataset/10_node_chain.pt',
                        help='directory to load graph data')
    parser.add_argument('--dataname', type=str, default='10_node_chain', help='data name')
    parser.add_argument('--savefolder', type=str, default='', help='save folder')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_dir', type=str, default='../result/Heisenberg/chain_8_node/GNNTrans/',
                        help='directory to save model parameter and energy list')
    parser.add_argument('--invariant', action='store_true', default=False, help="whether use invariant")
    parser.add_argument('--GPU', action='store_true', default=False, help="whether use GPU or not")
    parser.add_argument('--test', action='store_true', default=False, help="test mode")
    parser.add_argument('--optim', type=str, choices=['energy', 'it_swo'], default='energy')
    args = parser.parse_args()

    print(args)

    seed = 29  # 10086

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### load data and get undirected unique edge index list
    global data
    datapath = os.path.join(args.data_dir, args.dataname + '.pt')
    #     print(datapath)
    data = torch.load(datapath)
    idx_list = get_undirected_idx_list(data, periodic=False, square=False)
    # print("Undirected edge number: ", len(idx_list))

    global num_spin
    num_spin = args.num_spin
    num_hidden = args.num_spin
    if args.J2 != 0:
        J2 = args.J2
    else:
        J2 = None

    ### get hamiltonian matrix and ground truth energy
    E_exact = None
    if num_spin <= 16:
        H = tensor_prod_graph_Heisenberg(data, n=num_spin)
        # e_vals, e_vecs = np.linalg.eigh(H)
        [energy, state] = eigsh(H, k=1, which='SA')
        E_exact = energy  # e_vals[0]
        print('Exact energy: {}'.format(E_exact))

    ### visualize the loss history
    energy_list, precision_list = [], []

    def _update_curve(energy, precision, save_dir):
        energy_list.append(energy)
        precision_list.append(precision)
        if len(energy_list) % 10 == 0:
            fig = plt.figure()
            plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
            # dashed line for exact energy
            if E_exact is not None:
                plt.axhline(E_exact, ls='--')
            #     plt.show()
            fig.savefig(save_dir + 'energy.png')
            plt.close(fig)

    params = {
        'num_spins': args.num_spin,
        'hidden': args.emb_dim,
        'dropout': args.drop_ratio,
        'depth': args.num_layers
    }

    if args.GPU:
        # device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")

    if args.invariant == True:
        ansatz = Model_twoway_invariant(data, num_spin, num_hidden, args.model)
    else:
        if args.model == 'graph-transformer':
            row, col = data.edge_index
            deg = degree(col, data['x'].size(0))
            edge_attr = torch.cat((deg[row].view(1, -1), deg[col].view(1, -1)), dim=0)
            data.edge_attr = torch.transpose(edge_attr, 0, 1)
            ansatz = GraphTransformer(data, max(deg).item(), data.num_nodes, num_layers_gnn=4 // 2,
                                      num_layers_tran=4 // 2, emb_dim=32,
                                      head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last",
                                      graph_pooling='sum', norm='batch')
        elif args.model == 'gnn-transformer':
            ansatz = GT(data.num_nodes, num_layers_gnn=4 // 2, num_layers_tran=4 // 2, emb_dim=32,
                        head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum',
                        norm='batch')
        elif args.model == 'gnn-transformer-sublattice':
            ansatz = GT_sublattice(data.num_nodes, num_layers_gnn=4 // 2, num_layers_tran=4 // 2, emb_dim=32,
                                   head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last",
                                   graph_pooling='sum', norm='batch')
        elif args.model == 'gnn-transformer-twoway':
            ansatz = GT_twoway(data.num_nodes, num_layers_gnn=4 // 2, num_layers_tran=4 // 2, emb_dim=32,
                               head_size=4, dropout_rate=0, attention_dropout_rate=0, JK="last", graph_pooling='sum',
                               norm='batch')
        elif args.model == 'GN':
            ansatz = GraphNet(data.num_nodes, num_node_features=1, num_edge_features=1, dropout=0, depth=6, hidden=128, device=device, sublattice=False)
        elif args.model == 'GN2':
            ansatz = GraphNet2(data.num_nodes, num_node_features=1, num_edge_features=1, dropout=0, depth=6, hidden=128, device=device)
        elif args.model == 'cnn2d-explore':
            ansatz = CNN2D_explore(data.num_nodes, num_hidden, filters_in=1, filters=3, kernel_size=3)
        elif args.model == 'cnn2d':
            ansatz = CNN2D(data.num_nodes, num_hidden, filters_in=1, filters=3, kernel_size=3)
        elif args.model == 'cnn2d2':
            ansatz = CNN2D2(data.num_nodes, num_hidden, filters_in=1, filters=3, kernel_size=3)
        elif args.model == 'cnn2d-se-hex':
            ansatz = CNN2D_SE_Hex(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se-hex-108':
            ansatz = CNN2D_SE_Hex_108(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se-hex-fcn':
            ansatz = CNN2D_SE_Hex_FCN(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se-kagome':
            ansatz = CNN2D_SE_Kagome(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se-kagome-108':
            ansatz = CNN2D_SE_Kagome_108(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se-hex-new':
            ansatz = CNN2D_SE_Hex_new(data.num_nodes, num_hidden, filters_in=1, filters=64, kernel_size=3)
        elif args.model == 'cnn2d-se':
            ansatz = CNN2D_SE(data.num_nodes, num_hidden, filters_in=1, filters=4, kernel_size=3)
        elif args.model == 'cnn2d-v2':
            ansatz = CNN2D_v2(data.num_nodes, num_hidden, filters_in=1, filters=16, kernel_size=3)
        elif args.model == 'cnn2d-sublattice':
            ansatz = CNN2D_sublattice(data.num_nodes, num_hidden, filters_in=3, filters=3, kernel_size=3, device=device)
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
                                               drop_ratio=0,
                                               JK='last',
                                               graph_pooling='sum',
                                               norm='layer')
        elif args.model == 'deepergcn-virtual-twoway':
            ansatz = DeeperGCN_Virtualnode_twoway(data.num_nodes,
                                                  num_layers=5,
                                                  emb_dim=32,
                                                  drop_ratio=0,
                                                  JK='last',
                                                  graph_pooling='sum',
                                                  norm='layer')
        elif args.model == 'deepergcn-virtual-sublattice':
            ansatz = DeeperGCN_Virtualnode_sublattice(data.num_nodes,
                                                      num_layers=5,
                                                      emb_dim=32,
                                                      drop_ratio=0,
                                                      JK='last',
                                                      graph_pooling='sum',
                                                      norm='layer')
        elif args.model == 'gine-res-virtual-two-way-sym':
            ansatz = GINEVirtual_twoway_sym(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-two-way':
            ansatz = GINEVirtual_twoway(data.num_nodes, data.num_node_features, data.num_edge_features, **params)
        elif args.model == 'gine-res-virtual-two-way-sublattice':
            ansatz = GINEVirtual_twoway_sublattice(data.num_nodes, data.num_node_features, data.num_edge_features,
                                                   **params)
        else:
            raise ValueError('Invalid model type')


    if args.optim == 'it_swo':
        energy_phi = heisenberg_loc_it_swo_J1_J2
    else:
        energy_phi = None

    # model = VMCKernel(data, heisenberg_loc_batch_fast, ansatz=ansatz.cuda())
    # model = VMCKernel(data, heisenberg_loc_batch_fast, ansatz=ansatz.float().to(device))
    model = VMCKernel(data, heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.float().to(device), energy_phi=energy_phi)

    if args.optim == 'it_swo':
        ansatz_phi = deepcopy(model.ansatz)
        #  ansatz_phi = CNN2D(data.num_nodes, num_hidden, filters_in=1, filters=3, kernel_size=3).float().cuda()

    optimizer = torch.optim.Adam(model.ansatz.parameters(), lr=args.lr, betas=(0.9, 0.99))

    scheduler = StepLR(optimizer, step_size=9000, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    ### load pretrain model
    # path = '../result/Heisenberg/chain_4_node/GNNTrans/model.pt'
    # model.ansatz.load_state_dict(torch.load(path))
    # checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataname, args.model, '')
    # checkpoint = torch.load(checkpoint_dir + 'checkpoint.pt')
    #
    # # ansatz = RBM_twoway(num_spin, num_hidden)
    # # ansatz = RBM(num_spin, num_hidden)
    # ansatz = Model_twoway_invariant(data, num_spin, num_hidden, args.model)
    # model = VMCKernel(data, heisenberg_loc, ansatz=ansatz)
    # model.ansatz.load_state_dict(checkpoint['model_state_dict'])

    # checkpoint = torch.load('../checkpoints/Heisenberg/6_6_square_lattice_pbc_J2_05_new_vmc/cnn2d-se/checkpoint.pt')
    # print(checkpoint['energy'])
    # ansatz.load_state_dict(checkpoint['model_state_dict'])
    # model_batch = VMCKernelBatch(data=data, energy_loc=heisenberg_loc_batch_fast, ansatz=ansatz.cuda())
    # model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.double().to(device))

    if args.test:
        # print("Testing using best checkpoint!")
        # start = time.time()
        # checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
        # checkpoint_path = checkpoint_dir + 'checkpoint_best.pt'
        # print("Best checkpoint path: ", checkpoint_path)
        # checkpoint = torch.load(checkpoint_path)
        # print("Training energy for best model: ", checkpoint['energy'])
        # ansatz.load_state_dict(checkpoint['model_state_dict'])
        # model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.float().to(device))
        # test(model=model, num_spin=num_spin, idx_list=idx_list, J2=J2, device=device)
        # print("Testing time: ", time.time() - start)

        print("Testing using last checkpoint!")
        start = time.time()
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
        checkpoint_path = checkpoint_dir + 'checkpoint.pt'
        print("Last checkpoint path: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        print("Training energy for last model: ", checkpoint['energy'])
        ansatz.load_state_dict(checkpoint['model_state_dict'])
        model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.float().to(device))
        test(model=model, num_spin=num_spin, idx_list=idx_list, J2=J2, device=device)
        print("Testing time: ", time.time() - start)

        return

    num_params = sum(p.numel() for p in model.ansatz.parameters())
    print(f'#Params: {num_params}')

    if args.log_dir != '':
        log_dir = os.path.join(args.log_dir, args.savefolder, args.model, '')
        writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    first_batch = True
    sample_batch = None
    step_total = 0
    lowest_energy = 1000
    save_best = False
    for epoch in range(args.epochs):
        steps = int(200000 / 200)
        for step in range(steps):
            step_total = step + epoch * steps
            if step_total > 0:
                first_batch = False

            if args.optim == 'energy':
                energy, precision, sample_batch = train(model, num_spin, idx_list, optimizer, J2, first_batch=first_batch, 
                                                        sample_batch=sample_batch, device=device)
            elif args.optim == 'it_swo':
                if step % 30 == 0:
                    ansatz_phi.load_state_dict(model.ansatz.state_dict())
                energy, precision, sample_batch = train_it_swo(model, num_spin, idx_list, optimizer, J2, first_batch=first_batch, 
                                                               sample_batch=sample_batch, ansatz_phi=ansatz_phi, device=device)

            t1 = time.time()
            # energy = energy / num_spin
            # print('Step %d, dE/|E| = %.4f, elapse = %.4f' % (step, -(energy - E_exact)/E_exact, t1-t0))
            if energy < lowest_energy:
                lowest_energy = energy
                save_best = True
            print('Step %d, E = %.4f, elapse = %.4f, lowest_E = %.4f' % (step_total, energy, t1 - t0, lowest_energy))
            t0 = time.time()

            if args.log_dir != '':
                if E_exact is None:
                    writer.add_scalars('energy', {'pred': energy}, step_total)
                else:
                    writer.add_scalars('energy', {'pred': energy, 'gt': E_exact}, step_total)

            if args.save_dir != '':
                save_dir = os.path.join(args.save_dir, args.savefolder, args.model, '')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                _update_curve(energy, precision, save_dir)
                energy_save_dir = os.path.join(save_dir, 'graph_energy_list.npy')
                precision_save_dir = os.path.join(save_dir, 'graph_precision_list.npy')
                np.save(energy_save_dir, energy_list)
                np.save(precision_save_dir, precision_list)

            if args.checkpoint_dir != '':
                checkpoint = {'step': step_total, 'model_state_dict': model.ansatz.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'energy': energy, 'num_params': num_params}

                checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(checkpoint, checkpoint_dir + 'checkpoint.pt')

                if save_best:
                    save_best = False
                    torch.save(checkpoint, checkpoint_dir + 'checkpoint_best.pt')

            if optimizer.param_groups[0]['lr'] >= 0.000001:
                scheduler.step()    


if __name__ == "__main__":
    main()
import os
import torch
from typing import Dict
from argparse import ArgumentParser
from deepxube.training import avi
from environments.qcircuit import QCircuit
from nnet.nnet_utils import load_nnet_config, save_nnet_config

# setting random seed to ensure reproducibility
torch.manual_seed(123)


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--nnet_config', type=str)
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--num_qubits', type=int, required=True)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--step_max', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--itrs_per_update', type=int, default=1000)
    parser.add_argument('--max_itrs', type=int, default=1e5)
    parser.add_argument('--greedy_update_step_max', type=int, default=5)
    parser.add_argument('--num_update_procs', type=int, default=5)
    args = parser.parse_args()
    
    # environment setup
    nnet_config: Dict = None
    if args.nnet_config:
        load_nnet_config(args.nnet_config)
    
    env = QCircuit(
        num_qubits=args.num_qubits,
        nnet_config=nnet_config,
        epsilon=args.epsilon,
    )

    if not os.path.exists(args.nnet_dir):
        os.mkdir(args.nnet_dir)
    save_nnet_config(env.nnet_config, os.path.join(args.nnet_dir, 'nnet_config.yaml'))

    # running approximate value iteration
    avi.train(
        env=env,
        nnet_dir=args.nnet_dir,
        step_max=args.step_max,
        batch_size=args.batch_size,
        itrs_per_update=args.itrs_per_update,
        max_itrs=args.max_itrs,
        greedy_update_step_max=args.greedy_update_step_max,
        num_update_procs=args.num_update_procs,
    )
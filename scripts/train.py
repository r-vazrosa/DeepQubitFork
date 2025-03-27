from argparse import ArgumentParser
from deepxube.training import avi

from environments import QCircuit


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--num_qubits', type=int, required=True)
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--step_max', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--itrs_per_update', type=int, default=1000)
    parser.add_argument('--max_itrs', type=int, default=1e5)
    parser.add_argument('--greedy_update_step_max', type=int, default=5)
    parser.add_argument('--num_update_procs', type=int, default=5)
    args = parser.parse_args()
    args_dict = vars(args)

    # environment setup
    env = QCircuit(num_qubits=args_dict['num_qubits'])
    del args_dict['num_qubits']

    # running approximate value iteration
    avi.train(env=env, **args_dict)
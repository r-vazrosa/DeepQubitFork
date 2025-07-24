import os
import numpy as np
from argparse import ArgumentParser

from environments.qcircuit import QCircuit
from utils.matrix_utils import *


def main():
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_qubits', type=int, required=True)
    parser.add_argument('-k', '--num_targets', type=int, required=True)
    parser.add_argument('-p', '--save_path', type=str, required=True)
    parser.add_argument('-s', '--max_steps', type=int, default=1000)
    parser.add_argument('-m', '--method', type=str, default='rvs')
    parser.add_argument('--txt', action='store_true')
    parser.add_argument('--numpy', action='store_true')
    args = parser.parse_args()

    unitaries: np.ndarray[np.complex128]
    if args.method == 'random_walk':
        # environment setup/generating matrices
        env = QCircuit(args.num_qubits)
        start_states = env.get_start_states(args.num_targets)
        num_steps = np.random.randint(args.max_steps, size=(args.num_targets,))
#        states_walk = env._random_walk(start_states, num_steps)
        states_walk = env._random_walk(start_states, [args.max_steps] * args.num_targets)
        unitaries = [x.unitary for x in states_walk]
    elif args.method == 'rvs':
        unitaries = [random_unitary(2**args.num_qubits) for _ in range(args.num_targets)]

    # saving unitaries to files
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for i, U in enumerate(unitaries):
        if args.txt:
            filename = os.path.join(args.save_path, '%s.txt' % i)
            save_matrix_to_file(U, filename)
        if args.numpy:
            filename = os.path.join(args.save_path, '%s.npy' % i)
            np.save(filename, U)

if __name__ == '__main__':
    main()

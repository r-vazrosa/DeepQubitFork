import os
from argparse import ArgumentParser

from environments.qcircuit import QCircuit
from utils.matrix_utils import *


def main():
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_qubits', type=int, required=True)
    parser.add_argument('-k', '--num_targets', type=int, required=True)
    parser.add_argument('-s', '--max_steps', type=int, default=1000)
    parser.add_argument('-p', '--save_path', type=str, required=True)
    args = parser.parse_args()

    # environment setup/generating matrices
    env = QCircuit(args.num_qubits)
    start_states = env.get_start_states(args.num_targets)
    num_steps = np.random.randint(args.max_steps, size=(args.num_targets,))
    states_walk = env._random_walk(start_states, num_steps)
    unitaries = [x.unitary for x in states_walk]

    # saving unitaries to files
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    for i, U in enumerate(unitaries):
        filename = os.path.join(args.save_path, '%s.txt' % i)
        save_matrix_to_file(U, filename)


if __name__ == '__main__':
    main()
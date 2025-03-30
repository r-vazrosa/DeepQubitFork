"""
Script that generates a uniformly distributed
random sample of unitary matrices for 'n' qubits
"""
import pickle
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict
from utils.matrix_utils import random_unitary


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--num_goals', type=int, required=True)
    parser.add_argument('--num_qubits', type=int, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    args = parser.parse_args()

    # generating random unitary matrices
    print('Generating %d random %d-qubit unitary matrices' % (args.num_goals, args.num_qubits))
    random_mats: List[np.ndarray[np.complex128]] = \
        [random_unitary(2**args.num_qubits) for _ in range(args.num_goals)]
    
    # saving data
    print('Saving data to `%s`' % args.save_file)
    save_data: Dict = {'unitaries': random_mats, 'num_qubits': args.num_qubits}
    with open(args.save_file, 'wb') as f:
        pickle.dump(save_data, f)
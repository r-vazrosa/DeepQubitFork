"""
Script that generates a uniformly distributed
random sample of unitary matrices for 'n' qubits
"""
import os
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
    parser.add_argument('--pkl_file', type=str, default=None)
    parser.add_argument('--txt_dir', type=str, default=None)
    args = parser.parse_args()

    # generating random unitary matrices
    print('Generating %d random %d-qubit unitary matrices' % (args.num_goals, args.num_qubits))
    random_mats: List[np.ndarray[np.complex128]] = \
        [random_unitary(2**args.num_qubits) for _ in range(args.num_goals)]
    
    # saving data
    if args.pkl_file:
        # saving data using pickle format (for A* search)
        print('Saving pkl data to `%s`' % args.pkl_file)
        save_data: Dict = {'unitaries': random_mats, 'num_qubits': args.num_qubits}
        with open(args.pkl_file, 'wb') as f:
            pickle.dump(save_data, f)
    
    if args.txt_dir:
        # saving data in text format (for Synthetiq)
        print('Saving txt data to `%s`' % args.txt_dir)
        if not os.path.exists(args.txt_dir):
            os.mkdir(args.txt_dir)

        for (i, mat) in enumerate(random_mats):
            file_str = ''
            file_str += 'random%d\n' % i
            file_str += '%d\n' % args.num_qubits
            for row in mat:
                for x in row:
                    file_str += '(%0.10f,%0.10f) ' % (np.real(x), np.imag(x))
                file_str += '\n'
            for row in mat:
                for x in row:
                    file_str += '1 '
                file_str += '\n'

            save_file_path: str = os.path.join(args.txt_dir, 'random%d.txt' % i)
            with open(save_file_path, 'w') as f:
                print(file_str, file=f)
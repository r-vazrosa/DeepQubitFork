import pickle
from typing import List, Dict
import numpy as np
from argparse import ArgumentParser
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.synthesis import generate_basic_approximations
from qiskit.quantum_info import Operator


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--goals_file', type=str, required=True)
    parser.add_argument('--recurs_degree', type=int, default=3)
    parser.add_argument('--approx_depth', type=int, default=5)
    args = parser.parse_args()

    # setting up Solovay-Kitaev algorithm
    basis = ['t', 'tdg', 'h']
    approx = generate_basic_approximations(basis, depth=args.approx_depth)
    skd = SolovayKitaevDecomposition(approx)
    
    # loading goal matrices
    data: Dict
    with open(args.goals_file, 'rb') as f:
        data = pickle.load(f)

    # running solovay-kitaev
    approx_circuits = [skd.run(x, args.recurs_degree) for x in data['unitaries']]
    breakpoint()

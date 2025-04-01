import pickle
from typing import List, Dict
import numpy as np
from argparse import ArgumentParser
from qiskit import QuantumCircuit
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.synthesis import generate_basic_approximations
from qiskit.quantum_info import Operator
from utils.matrix_utils import unitary_distance


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--goals_file', type=str, required=True)
    parser.add_argument('--recursion_degree', type=int, default=3)
    parser.add_argument('--approximation_depth', type=int, default=3)
    args = parser.parse_args()

    # loading goals
    data: Dict
    with open(args.goals_file, 'rb') as f:
        data = pickle.load(f)

    num_qubits: int = data['num_qubits']
    unitaries: List[np.ndarray[np.complex128]] = data['unitaries']
    
    circuits: List[QuantumCircuit] = []
    for x in unitaries:
        circ = QuantumCircuit(num_qubits)
        circ.unitary(x, qubits=list(range(num_qubits)))
        circuits.append(circ)

    # running compilation
    print('Compiling %d unitaries to discrete approximations' % len(unitaries))
    approx_circuits: List[QuantumCircuit] = []
    if num_qubits == 1:
        # running Solovay-Kitaev on one-qubit unitaries
        basis_gates = ['h', 't', 'tdg', 's', 'sdg']
        approx = generate_basic_approximations(basis_gates, depth=args.approximation_depth)
        skd = SolovayKitaev(recursion_degree=args.recursion_degree, basic_approximations=approx)
        approx_circuits = [skd(x) for x in circuits]
    
    else:
        # TODO: implement synthesis for more than 1 qubit
        pass

    # printing out info about compilation
    gate_depths: List[int] = [x.depth() for x in approx_circuits]
    errors: List[float] = [unitary_distance(Operator(x).data, y) \
                           for (x, y) in zip(approx_circuits, unitaries)]
    
    print('Done')
    print('Gate count (min/max/mean): %d/%d/%.2f' % (min(gate_depths), max(gate_depths), \
                                                     sum(gate_depths) / len(gate_depths)))
    print('Errors: (min/max/mean): %.2f/%.2f/%.2f' % (min(errors), max(errors), \
                                                      sum(errors) / len(errors)))
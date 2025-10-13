import scipy
from scipy.linalg import schur
import numpy as np
from numpy import trace, log, exp, diag, diagonal
from numpy.linalg import det, eig, inv, svd
from typing import List
from qiskit import qasm2
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.circuit.library import U3Gate


# identity matrix on one qubit
I = np.eye(2, dtype=np.complex128)
# 'zero' project for one qubit
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# 'one' projector for one qubit
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def load_matrix_from_file(filename: str) -> np.ndarray[np.complex128]:
    if filename.endswith('.txt'):
        num_qubits: int
        matrix: np.ndarray[np.complex128]
        with open(filename, 'r') as f:
            lines = [x.strip() for x in list(f)]
            num_qubits = int(lines[1])
            N = 2**(num_qubits)
            matrix = np.zeros((N, N), dtype=np.complex128)
            for i in range(N):
                row = lines[2+i]
                cols = row.split(' ')
                for j, col in enumerate(cols):
                    left, right = col.split(',')
                    real = float(left[1:])
                    imag = float(right[:-1])
                    matrix[i][j] = real + imag*1j
        return num_qubits, matrix
    
    elif filename.endswith('.npy'):
        matrix = np.load(filename)
        num_qubits = int(np.log2(matrix.shape[0]))
        return num_qubits, matrix

    else:
        raise Exception('Invalid file format')


def save_matrix_to_file(matrix: np.ndarray[np.complex128], filename: str):
    num_qubits = int(np.log2(matrix.shape[0]))
    with open(filename, 'w') as f:
        f.write('matrix\n%s' % num_qubits)
        for row in matrix:
            row_str = '\n'
            for x in row:
                row_str += '(%s,%s) ' % (np.real(x), np.imag(x))
            f.write(row_str)


def qasm_to_matrix(qasm_str: str) -> np.ndarray[np.complex128]:
    return Operator(qasm2.loads(qasm_str)).data


def seq_to_matrix(seq: str) -> np.ndarray[np.complex128]:
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg qs[1];"""

    for x in seq:
        qasm_str += '\n' + x + ' qs[0];'
    
    qc = qasm2.loads(qasm_str)
    op = Operator.from_circuit(qc)
    return op.data


def tensor_product(mats: List[np.ndarray[np.complex128]]) -> np.ndarray[np.complex128]:
    """Computes the tensor product (Kronecker product) of a list of matrices"""
    current = 1
    for mat in mats:
        current = np.kron(current, mat)
    return current


def phase_align(U: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    """Aligns the global phase of a unitary so that
       phase_align(U) == phase_align(W) if U=e^(i*theta)W"""
    N = U.shape[0]
    mu = (1/(N**2)) * np.sum(U ** 2)
    if mu == 0.0:
        mu = 1.0
    mu_norm = mu / np.abs(mu)
    mu_half = mu_norm * np.exp(-1j*np.angle(mu_norm)/2)
    mu_conj = np.conj(mu_half)
    W = mu_conj * U
    if np.real(W[0][0]) < 0:
        W = np.exp(1j*np.pi) * W
    return W


def hash_unitary(unitary: np.ndarray[np.complex128], tolerance: float = 0.001) -> int:
    """Creates fixed-length representation of unitary operator"""
    return hash(tuple(np.round(phase_align(unitary).flatten() / tolerance)))


def unitary_distance(U: np.ndarray[np.complex128], C: np.ndarray[np.complex128]) -> float:
    """Computes the distance between two unitaries"""
    # from paper 'Synthetiq: Fast and Versatile Quantum Circuit Synthesis'
    M = np.ones(U.shape, dtype=np.complex128)
    tr_cu = np.trace(np.matmul(invert_unitary(M * C), M * U))
    if tr_cu == 0.: tr_cu = 1.
    num = np.linalg.norm(M * U - (tr_cu / np.abs(tr_cu)) * M * C)
    d_sc = num / np.sqrt(np.linalg.norm(M))
    return d_sc


def random_unitary(dim: int) -> np.ndarray[np.complex128]:
    """Generates a randomly distributed set of unitary matrices"""
    return scipy.stats.unitary_group.rvs(dim)


def invert_unitary(U: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    """Inverts a unitary matrix"""
    return U.conj().T

import scipy
import numpy as np
from typing import List
from qiskit import qasm3
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


def seq_to_matrix(seq: str) -> np.ndarray[np.complex128]:
    qasm_str = '''
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit q;'''

    for x in seq:
        qasm_str += '\n' + x + ' q;'
    
    qc = qasm3.loads(qasm_str)
    op = Operator.from_circuit(qc)
    return op.data


def gen_u3(theta, phi, _lambda):
    return Operator(U3Gate(theta, phi, _lambda)).data.astype(np.complex128)


def gram_schmidt(A):
    """input: A set of linearly independent vectors stored
              as the columns of matrix A
       outpt: An orthongonal basis for the column space of A.
       (Copied from https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/)"""
    # get the number of vectors.
    A = np.copy(A) # create a local instance of the array
    n = A.shape[1]
    for j in range(n):
        # For the vector in column j, find the perpendicular
        # of the projection onto the previous orthogonal vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        # If original vectors aren't lin indep then we can check for this:
        # 

        if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
            A[:, j] = np.zeros(A.shape[0])
        else:    
            A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A


def perturb_unitary(U: np.ndarray[np.complex128], epsilon: float):
    """Adds a small perturbation to a unitary matrix
       such that it is still within epsilon of the original"""
    # N = U.shape[0]
    # random_matrix = ((np.random.rand(N,N)*2-1) + (np.random.rand(N,N)*2-1)*1j) * epsilon / 32
    # U_new = gram_schmidt(U + random_matrix)
    # return U_new
    U_eps = gen_u3(np.random.rand(3)*2*np.pi*1e-4)
    U_new = U_eps @ U
    assert unitary_distance(U, U_new) <= epsilon
    return U_new


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


def unitary_to_nnet_input(U: np.ndarray[np.complex128]) -> np.ndarray[float]:
    """Converts a complex-valued unitary matrix into real-valued
       flat numpy arrays that can be converted to tensors easily"""
    # splitting array into magnitude and phase of each complex entry
    # W = np.array([np.abs(U), np.angle(U)/(2*np.pi)]).flatten()
    # return W
    qc = OneQubitEulerDecomposer(basis='U3')(U)
    if len(qc.data) > 0:
        return np.array(qc.data[0].params) / (2*np.pi)
    else:
        return np.array([0.0, 0.0, 0.0])


def unitary_distance(U: np.ndarray[np.complex128], C: np.ndarray[np.complex128]) -> float:
    """Computes the distance between two unitaries"""
    trc = np.trace( np.matmul(U, C.conj().T) )
    inner = (1/(2**(U.shape[0]))) * np.abs( trc )**2
    if inner > 1.0:
        # sometimes small rounding errors can occur and cause this value
        #  to be > 1, which causes a sqrt of a negative number error
        inner = 1.0
    return np.sqrt( 1.0 - inner )


def random_unitary(dim: int) -> np.ndarray[np.complex128]:
    """Generates a randomly distributed set of unitary matrices"""
    return scipy.stats.unitary_group.rvs(dim)


def invert_unitary(U: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    """Inverts a unitary matrix"""
    return U.conj().T

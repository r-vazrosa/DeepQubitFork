import scipy
import numpy as np
from typing import List
from qiskit import qasm3
from qiskit.quantum_info import Operator


# identity matrix on one qubit
I = np.eye(2, dtype=np.complex128)
# 'zero' project for one qubit
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# 'one' projector for one qubit
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def load_matrix_from_file(filename: str) -> np.ndarray[np.complex128]:
    num_qubits: int
    unitary: np.ndarray[np.complex128]
    with open(filename, 'r') as f:
        lines = [x.strip() for x in list(f)]
        num_qubits = int(lines[1])
        N = 2**(num_qubits)
        unitary = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            row = lines[2+i]
            cols = row.split(' ')
            for j, col in enumerate(cols):
                left, right = col.split(',')
                real = float(left[1:])
                imag = float(right[:-1])
                unitary[i][j] = real + imag*1j
    return num_qubits, unitary


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


def gram_schmidt(A):
    '''input: A set of linearly independent vectors stored
              as the columns of matrix A
       outpt: An orthongonal basis for the column space of A.
       (Copied almost entirely from https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/)
    '''
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
    # N = U.shape[0]
    # random_matrix = np.random.rand(N,N) + 1j*np.random.rand(N,N)
    # random_matrix = (random_matrix + random_matrix.conj().T) / 2
    # perturbation = random_matrix / np.linalg.norm(random_matrix) * epsilon

    # U_perturbed = U + perturbation
    # W, _, Vh = np.linalg.svd(U_perturbed)
    # U_new = W @ Vh
    # return U_new
    N = U.shape[0]
    random_matrix = np.random.rand(N,N) * epsilon
    U_new = gram_schmidt(U + random_matrix)
    return U_new


def tensor_product(mats: List[np.ndarray[np.complex128]]) -> np.ndarray[np.complex128]:
    """
    Computes the tensor product (Kronecker product) of a list of matrices

    @param mats: List of numpy complex matrices
    @returns: Numpy complex matrix result of tensor product
    """
    current = 1
    for mat in mats:
        current = np.kron(current, mat)
    return current


def phase_align(unitary: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    """
    Aligns the global phase of a unitary so that the top left
    element has complex phase 0

    @param unitary: n x n unitary matrix to align
    @returns: n x n re-aligned unitary matrix
    """
    phs: float = np.angle(unitary[0][0])
    return np.exp(-1j * phs) * unitary


def hash_unitary(unitary: np.ndarray[np.complex128], tolerance: float = 0.001) -> int:
    """
    Creates fixed-length representation of unitary operator

    @param unitary: n x n unitary matrix
    @param tolerance: Level of discretization for matrix values
    @returns: Integer uniquely representing matrix up to tolerance
    """
    return hash(tuple(np.round(phase_align(unitary).flatten() / tolerance)))


def unitary_to_nnet_input(unitary: np.ndarray[np.complex128], L: int) -> np.ndarray[float]:
    """
    Converts a complex-valued unitary matrix into real-valued
    flat numpy arrays that can be converted to tensors easily

    @param unitary: Unitary matrix to convert
    @returns: Numpy vector of real and imaginary values of matrix
    """
    # unitary_aligned = phase_align(unitary)
    # unitary_flat = unitary_aligned.flatten()
    # unitary_real = np.real(unitary_flat)
    # unitary_imag = np.imag(unitary_flat)
    # unitary_nnet = np.hstack((unitary_real, unitary_imag)).astype(float)
    # return unitary_nnet

    # global-phase invariant transformation
    # from Making Neural Networks More Suitable for Approximate Clifford+T Circuit Synthesis (Weiden, 2025)
    N = unitary.shape[0]
    mu = (1/(N**2)) * np.sum(unitary ** 2)
    if mu == 0.0:
        mu = 1.0
    mu_norm = mu / np.abs(mu)
    mu_half = mu_norm * np.exp(-1j*np.angle(mu_norm)/2)
    mu_conj = np.conj(mu_half)
    W = mu_conj * unitary
    if np.real(W[0][0]) < 0:
        W = np.exp(1j*np.pi) * W

    # neural radiance field encoding
    # from NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall, 2022)
#    omegas = 2**(np.arange(L)) * np.pi / 2
#    x = np.array([np.real(W), np.imag(W)])
#    x1 = np.matmul(x.reshape(-1, 1), omegas.reshape(1, -1))
#    gamma_sin = np.sin(x1)
#    gamma_cos = np.cos(x1)
#    gamma = np.hstack([gamma_sin.flatten(), gamma_cos.flatten()])
#    return gamma
    x = 2**(np.arange(L))
    y = np.array([np.real(unitary), np.imag(unitary)])
    x1 = np.matmul(x.reshape(-1, 1), y.reshape(1, -1)).flatten()
    x2 = x1 - np.trunc(x1) * ((x1 > 1.0).astype(int) | (x1 < -1.0).astype(int))
    return x2


def unitary_distance(U: np.ndarray[np.complex128], C: np.ndarray[np.complex128]) -> float:
    """
    Computes the distance between two matrices using the operator norm

    @param mat1: First unitary
    @param mat2: Second unitary
    @param method: Which version of the distance function to use
    @returns: Distance as floating point number
    """
    # from paper 'Synthetiq: Fast and Versatile Quantum Circuit Synthesis'
    # M = np.ones(U.shape, dtype=np.complex128)
    # tr_cu = np.trace(np.matmul(invert_unitary(M * C), M * U))
    # if tr_cu == 0.: tr_cu = 1.
    # num = np.linalg.norm(M * U - (tr_cu / np.abs(tr_cu)) * M * C)
    # d_sc = num / np.sqrt(np.linalg.norm(M))
    # return d_sc
    trc = np.trace( np.matmul(U, C.conj().T) )
    inner = (1/(2**(U.shape[0]))) * np.abs( trc )**2
    return np.sqrt( np.around( 1.0 - inner , 10 ) )


def random_unitary(dim: int) -> np.ndarray[np.complex128]:
    """
    Generates a randomly distributed set of unitary matrices

    @param dim: Dimension of unitary group to generate
    @return: Numpy complex array of unitary matrix
    """
    return scipy.stats.unitary_group.rvs(dim)


def invert_unitary(unitary: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    """
    Inverts a unitary matrix

    @param unitary: Numpy complex matrix to invert
    @returns: Inverted numpy complex matrix
    """
    return np.conj(unitary.T)

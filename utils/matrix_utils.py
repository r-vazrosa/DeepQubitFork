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
    U_eps = gen_u3(*((np.random.rand(3)-0.5)*np.pi*5e-4))
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
        return np.array(qc.data[0].params) / np.pi
    else:
        return np.array([0.0, 0.0, 0.0])


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


def get_gell_mann_basis(n: int):
    matrices = []
    for k in range(n):
        for j in range(k):
            S = np.zeros((n,n), dtype=np.complex64)
            S[j,k] = 1
            S[k,j] = 1
            matrices.append(S)

            A = np.zeros((n,n), dtype=np.complex64)
            A[j,k] = -1j
            A[k,j] = 1j
            matrices.append(A)

    for m in range(n-1):
        D = np.zeros((n,n), dtype=np.complex64)
        for j in range(m+1):
            D[j,j] = 1
            D[m+1,m+1] = -1*(m+1)
        D = D * np.sqrt(2 / ((m+2)*(m+1)))
        matrices.append(D)

    return np.stack(matrices)


def project_to_unitary(U):
    # Reorthogonalize U using SVD
    Uu, _, Uv = svd(U)
    return Uu @ Uv


def logm_unitary_eig(U):
    eigvals, eigvecs = eig(U)
    # log_eigvals = log(eigvals)
    # D_log = diag(log_eigvals)

    eigvals /= np.abs(eigvals)
    angles = np.angle(eigvals)
    log_diag = 1j * angles
    D_log = diag(log_diag)
    
    V = eigvecs
    V_inv = inv(V)
    log_U = V @ D_log @ V_inv
    return log_U


def logm_unitary_schur(U):
    T, Q = schur(U, output='complex')
    log_diag = log(diagonal(T))

    log_T = diag(log_diag)
    logU = Q @ log_T @ Q.conj().T
    logU = 0.5 * (logU - logU.conj().T)
    return logU


def logm_unitary(U):
    U = project_to_unitary(U)
    try:
        return logm_unitary_eig(U)
    except np.linalg.LinAlgError:
        return logm_unitary_schur(U)
    except np.linalg.LinAlgError:
        n = U.shape[0]
        pert = 1e-10 * np.eye(n, dtype=U.dtype)
        return logm_unitary(U + pert)


def gell_mann_encoding(U, generators):
    if len(U.shape) < 3:
        n = U.shape[0]
        U = U.reshape(1, n, n)
        
    # computing determinant
    det_U = det(U)

    # extracting global phase
    n = U.shape[-2]
    phi = np.angle(det_U) / n
    phi = phi[:, np.newaxis, np.newaxis]
    U_hat = exp(-1j*phi) * U

    # taking matrix log
    log_U = []
    for u in U_hat:
        log_u = logm_unitary(u)
        log_U.append(log_u)

    log_U = np.array(log_U)
    H = -1j * log_U

    # constructing angles
    thetas = []
    for G in generators:
        theta_a = trace(H @ G, axis1=1, axis2=2)
        theta_a = np.real(theta_a)
        thetas.append(theta_a)

    return (np.vstack(thetas) / 2).T
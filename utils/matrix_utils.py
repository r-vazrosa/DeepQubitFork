import scipy
import numpy as np
from typing import List


# identity matrix on 2 qubits
I = np.eye(2, dtype=np.complex128)
# 'zero' project for one qubit
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# 'one' projector for one qubit
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


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


def unitary_to_nnet_input(unitary: np.ndarray[np.complex128]) -> np.ndarray[float]:
    """
    Converts a complex-valued unitary matrix into real-valued
    flat numpy arrays that can be converted to tensors easily

    @param unitary: Unitary matrix to convert
    @returns: Numpy vector of real and imaginary values of matrix
    """
    unitary_flat = unitary.flatten()
    unitary_real = np.real(unitary_flat)
    unitary_imag = np.imag(unitary_flat)
    unitary_nnet = np.hstack((unitary_real, unitary_imag)).astype(float)
    return unitary_nnet


def mats_close(mat1: np.ndarray[np.complex128], mat2: np.ndarray[np.complex128], epsilon: float) -> bool:
    """
    Computes the distance between two matrices using the operator norm
    return whether they are within a certain tolerance 'epsilon'

    @param epsilon: Decimal number representing error tolerance
    @returns: Whether the two matrices are close
    """
    return np.linalg.norm(mat1 - mat2) <= epsilon


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
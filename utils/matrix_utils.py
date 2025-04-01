import scipy
import numpy as np
from typing import List


# identity matrix on one qubit
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
    @returns: Integer uniquely representing matrix up to global phase and tolerance
    """
    return hash(tuple(np.round(phase_align(unitary).flatten() / tolerance)))


def unitary_to_nnet_input(unitary: np.ndarray[np.complex128]) -> np.ndarray[float]:
    """
    Converts a complex-valued unitary matrix into real-valued
    flat numpy arrays that can be converted to tensors easily

    @param unitary: Unitary matrix to convert
    @returns: Numpy vector of real and imaginary values of matrix
    """
    unitary_aligned = phase_align(unitary)
    unitary_flat = unitary_aligned.flatten()
    unitary_real = np.real(unitary_flat)
    unitary_imag = np.imag(unitary_flat)
    unitary_nnet = np.hstack((unitary_real, unitary_imag)).astype(float)
    return unitary_nnet


def unitary_distance(mat1: np.ndarray[np.complex128], mat2: np.ndarray[np.complex128]) -> float:
    """
    Computes the distance between two matrices using the operator norm

    Uses `phase_align` to make sure that a difference in the global phase
    of the two unitaries is removed and does not return false negatives

    @param mat1: First unitary
    @param mat2: Second unitary
    @returns: Distance as floating point number
    """
    return np.linalg.norm(phase_align(mat1) - phase_align(mat2))


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
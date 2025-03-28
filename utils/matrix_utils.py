import numpy as np
from typing import List


I = np.eye(2, dtype=np.complex64)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex64)


def tensor_product(mats: List[np.ndarray[np.complex64]]) -> np.ndarray[np.complex64]:
    current = 1
    for mat in mats:
        current = np.kron(current, mat)
    return current


def unitary_to_nnet_input(unitary: np.ndarray[np.complex64]) -> np.ndarray[float]:
    unitary_flat = unitary.flatten()
    unitary_real = np.real(unitary_flat)
    unitary_imag = np.imag(unitary_flat)
    unitary_nnet = np.hstack((unitary_real, unitary_imag)).astype(float)
    return unitary_nnet


def mats_close(mat1: np.ndarray[np.complex64], mat2: np.ndarray[np.complex64]) -> bool:
    return np.allclose(mat1, mat2, rtol=1e-5, atol=1e-6)
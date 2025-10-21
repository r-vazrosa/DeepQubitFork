import numpy as np
from utils.matrix_utils import load_matrix_from_file, save_matrix_to_file

matrix = np.array(
    [[1+0j, 0+1j],
     [0-1j, 2+3j]],
                  
    dtype=np.complex128
)

mask = np.array(
    [[1, 0],
     [0, 1]],
    dtype=np.uint8
)

save_matrix_to_file(matrix, mask, "tests/matrix_test.txt")
num_qubits, load_matrix, load_mask = load_matrix_from_file("tests/matrix_test.txt")

print("num_qubits")
print(load_matrix)
print(load_mask)
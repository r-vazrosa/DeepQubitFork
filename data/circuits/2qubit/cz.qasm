// Gate-count: 3, T-count: 0, time: 0.518s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
h qubits[0];
cx qubits[1], qubits[0];
h qubits[0];

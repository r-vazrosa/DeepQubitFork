// Gate-count: 15, T-count: 7, time: 112.374s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[3];
cx qubits[0], qubits[1];
t qubits[1];
h qubits[2];
cx qubits[2], qubits[1];
tdg qubits[1];
cx qubits[0], qubits[1];
cx qubits[2], qubits[0];
t qubits[0];
cx qubits[2], qubits[0];
t qubits[1];
cx qubits[2], qubits[1];
tdg qubits[2];
h qubits[2];
tdg qubits[0];
tdg qubits[1];

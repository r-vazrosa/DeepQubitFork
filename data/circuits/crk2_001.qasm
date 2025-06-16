// Gate-count: 5, T-count: 3, time: 2.316s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
t qubits[0];
cx qubits[1], qubits[0];
tdg qubits[0];
cx qubits[1], qubits[0];
t qubits[1];

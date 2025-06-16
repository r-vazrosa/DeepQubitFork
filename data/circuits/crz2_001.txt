// Gate-count: 4, T-count: 2, time: 1.369s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
cx qubits[1], qubits[0];
tdg qubits[0];
cx qubits[1], qubits[0];
t qubits[1];

// Gate-count: 7, T-count: 2, time: 4.134s
OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[2];
sdg qubits[0];
h qubits[0];
tdg qubits[0];
cx qubits[1], qubits[0];
t qubits[0];
h qubits[0];
s qubits[0];

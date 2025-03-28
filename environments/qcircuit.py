import torch
from torch import nn
import numpy as np
from hashlib import sha256
from typing import Self, Tuple, List
from deepxube.environments.environment_abstract import Environment, State, Action, Goal, HeurFnNNet
from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel
from utils.matrix_utils import *


class QState(State):
    def __init__(self, unitary: np.ndarray[np.complex64] = None):
        self.unitary = unitary
    
    def __hash__(self):
        return hash(sha256(self.unitary.tobytes()).hexdigest())

    def __eq__(self, other: Self):
        return mats_close(self.unitary, other.unitary)


class QGoal(Goal):
    def __init__(self, unitary: np.ndarray[np.complex64]):
        self.unitary = unitary
    

class QAction(Action):
    def __init__(self, unitary: np.ndarray[np.complex64]):
        self.unitary = unitary

    def apply_to(self, state: QState) -> QState:
        new_state_unitary = np.matmul(self.full_gate_unitary, state.unitary)
        return QState(new_state_unitary)


class OneQubitGate(QAction):
    def __init__(self, num_qubits: int, qubit: int, unitary: np.ndarray[np.complex64]):
        super(OneQubitGate, self).__init__(unitary)
        self.qubit = qubit
        self.num_qubits = num_qubits
        self.unitary = unitary
        self._generate_full_unitary()
    
    def _generate_full_unitary(self):
        mats: List[np.ndarray] = []
        for i in range(self.num_qubits):
            if i == self.qubit:
                mats.append(self.unitary)
            else:
                mats.append(I)
        
        self.full_gate_unitary = tensor_product(mats)
    

class ControlledGate(QAction):
    def __init__(self, num_qubits: int, control: int, target: int, unitary: np.ndarray[np.complex64]):
        super(ControlledGate, self).__init__(unitary)
        self.control = control
        self.target = target
        self.num_qubits = num_qubits
        self.unitary = unitary
        self._generate_full_unitary()

    def _generate_full_unitary(self):
        p0_mats = [] # matrices that are applied to each qubit when the control is 0...
        p1_mats = [] # ...and when the control is 1
        for i in range(self.num_qubits):
            if i == self.control:
                p0_mats.append(P0)
                p1_mats.append(P1)
            elif i == self.target:
                p0_mats.append(I)
                p1_mats.append(self.unitary)
            else:
                p0_mats.append(I)
                p1_mats.append(I)
        
        p0_full = tensor_product(p0_mats)
        p1_full = tensor_product(p1_mats)
        self.full_gate_unitary = p0_full + p1_full

    
class HGate(OneQubitGate):
    unitary = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)

    def __init__(self, num_qubits: int, qubit: int):
        super(HGate, self).__init__(num_qubits, qubit, self.unitary)

    
class SGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

    def __init__(self, num_qubits: int, qubit: int):
        super(SGate, self).__init__(num_qubits, qubit, self.unitary)

    
class TGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex64)

    def __init__(self, num_qubits: int, qubit: int):
        super(TGate, self).__init__(num_qubits, qubit, self.unitary)


class CNOTGate(ControlledGate):
    unitary = np.array([[0, 1], [1, 0]], dtype=np.complex64)

    def __init__(self, num_qubits: int, control: int, target: int):
        super(CNOTGate, self).__init__(num_qubits, control, target, self.unitary)


class QNNet(HeurFnNNet):
    def __init__(self, input_size: int, resnet_dim: int, num_resnet_blocks: int, \
                 fc_input_dim: int, fc_layer_dims: List[int]):
        super(QNNet, self).__init__(nnet_type='V')
        
        self.fc_input = nn.Linear(input_size, resnet_dim)
        self.resnet = ResnetModel(resnet_dim, num_resnet_blocks, fc_input_dim, batch_norm=False)
        self.fully_connected = FullyConnectedModel(
            fc_input_dim,
            fc_layer_dims,
            layer_batch_norms=[False] * len(fc_layer_dims),
            layer_acts=['RELU'] * len(fc_layer_dims),
        )
    
    def forward(self, states_goals_l: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(states_goals_l, dim=1).float()
        x = self.fc_input(x)
        x = self.resnet(x)
        x = self.fully_connected(x)
        return x


class QCircuit(Environment):
    gate_set = [HGate, SGate, TGate, CNOTGate]

    def __init__(self, num_qubits: int):
        super(QCircuit, self).__init__(env_name='qcircuit')
        
        self.start_walk_max: int = 100
        self.num_qubits: int = num_qubits
        self._generate_actions()

    def _generate_actions(self):
        """
        Generates the action set for n qubits given a specific gate set
        by looping over each possible gate at each qubit
        """
        self.actions = []
        for i in range(self.num_qubits):
            for gate in self.gate_set:
                if issubclass(gate, OneQubitGate):
                    for i in range(self.num_qubits):
                        self.actions.append(gate(self.num_qubits, i))
                elif issubclass(gate, ControlledGate):
                    for i in range(self.num_qubits):
                        for j in range(self.num_qubits):
                            if i != j:
                                self.actions.append(gate(self.num_qubits, i, j))

    def get_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with random unitary operators initialized

        @param num_states: Number of states to generate
        @returns: Generated states

        TODO: look into more sophisticated way of randomly generating unitary matrices,
        such as http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
        """
        states = [QState(np.eye(2**self.num_qubits, dtype=np.complex64)) for _ in range(num_states)]
        num_walk_steps = np.random.randint(self.start_walk_max, size=(len(states,)))
        states_walk = self._random_walk(states, num_walk_steps)
        return states_walk

    def get_state_actions(self, states: List[QState]) -> List[List[QAction]]:
        return [[x for x in self.actions] for _ in states]

    def next_state(self, states: List[QState], actions: List[QAction]) -> Tuple[List[QState], List[float]]:
        next_states = []
        for state, action in zip(states, actions):
            next_state = action.apply_to(state)
            next_states.append(next_state)

        transition_costs = [1.0] * len(states)
        return next_states, transition_costs

    def sample_goal(self, states_start: List[QState], states_goal: List[QState]) -> List[QGoal]:
        return [QGoal(x.unitary) for x in states_goal]
    
    def is_solved(self, states: List[QState], goals: List[QGoal]) -> List[bool]:
        """
        Checks whether each state is solved by comparing their unitaries (within a tolerance)

        @param states: List of quantum circuit states
        @param goals: List of goals to check against
        @returns: List of bools representing solved/not-solved

        TODO: Implement more sophisticated way of checking of two
        unitaries are 'close', possibly using some kind of matrix norm
        """
        return [mats_close(state.unitary, goal.unitary) for (state, goal) in zip(states, goals)]

    def states_goals_to_nnet_input(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray[float]]:
        """
        Converts quantum state class objects to numpy arrays that can be
        converted to tensors for neural network training

        @param states: List of quantum circuit states
        @param goals: List of quantum circuit goals
        @returns: List of numpy arrays of flattened state and unitaries (in float format)
        """
        states_nnet = np.vstack([unitary_to_nnet_input(x.unitary) for x in states])
        goals_nnet = np.vstack([unitary_to_nnet_input(x.unitary) for x in goals])
        return [states_nnet, goals_nnet]

    def get_v_nnet(self) -> HeurFnNNet:
        input_size: int = (2**(2*self.num_qubits + 2))
        return QNNet(input_size, 2000, 4, 400, [200, 100, 80, 20, 1])

    def get_q_nnet(self) -> HeurFnNNet:
        raise NotImplementedError()
    
    def get_pddl_domain(self):
        raise NotImplementedError()
    
    def state_goal_to_pddl_inst(self, state, goal):
        raise NotImplementedError()
    
    def pddl_action_to_action(self, pddl_action):
        raise NotImplementedError()
    
    def visualize(self, states, goals):
        raise NotImplementedError()
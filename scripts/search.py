"""
Script that uses A* search to synthesize a
unitary matrix from an arbitrary gate set
"""
import os
import pickle
from argparse import ArgumentParser
import numpy as np
from typing import Dict, List, Optional
from deepxube.search.astar import AStar, get_path
from deepxube.nnet import nnet_utils
from environments.qcircuit import QCircuit, QGoal, QState, QAction
from utils.matrix_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--goal_file', type=str, required=True)
    parser.add_argument('--goal_format', type=str, default='pkl')
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-2)
    parser.add_argument('--path_weight', type=float, default=0.2)
    args = parser.parse_args()

    # loading goal data
    data: Dict = dict()
    if args.goal_format == 'pkl':
        with open(args.goal_file, 'rb') as f:
            data = pickle.load(f)
    
    elif args.goal_format == 'txt':
        with open(args.goal_file, 'r') as f:
            lines = [x.strip() for x in list(f)]
            data['num_qubits'] = int(lines[1])
            N = 2**(data['num_qubits'])
            matrix = np.zeros((N, N), dtype=np.complex128)
            for i in range(N):
                row = lines[2+i]
                cols = row.split(' ')
                for j, col in enumerate(cols):
                    left, right = col.split(',')
                    real = float(left[1:])
                    imag = float(right[:-1])
                    matrix[i][j] = real + imag*1j
                    data['unitaries'] = [matrix]

    else:
        raise Exception('Invalid goal file format `%s`' % args.goal_format)

    # environment setup
    env: QCircuit = QCircuit(num_qubits=data['num_qubits'], epsilon=args.epsilon)
    goals: List[QGoal] = [QGoal(x) for x in data['unitaries']]
    start_states: List[QState] = [QState(tensor_product([I] * data['num_qubits'])) for _ in goals]
    weights: List[float] = [args.path_weight] * len(start_states)

    # loading heuristic function
    device, devices, on_gpu = nnet_utils.get_device()
    nnet_weights_file: str = os.path.join(args.nnet_dir, 'target.pt')
    heuristic_fn = nnet_utils.load_heuristic_fn(nnet_weights_file, device, on_gpu, env.get_v_nnet(), env)

    # setup A* search
    astar = AStar(env)
    astar.add_instances(start_states, goals, weights, heuristic_fn)

    # running search
    step: int = 0
    while np.any([not x.finished for x in astar.instances]) and step < args.max_steps:
        astar.step(heuristic_fn, args.batch_size, verbose=True)
        print('Solved: %d/%d\n' % (sum([int(x.finished) for x in astar.instances]), len(goals)))
        step += 1
    
    # getting paths
    paths: List[List[Optional[QAction]]] = []
    path_lens: List[int] = []
    for x in astar.instances:
        if x.finished:
            _, path_actions, _ = get_path(x.goal_node)
            paths.append(path_actions)
            path_lens.append(len(path_actions))
        else:
            paths.append(None)

    if len(path_lens) == 0:
        print('Could not find any paths')
    
    else:
        # saving paths
        print('Found %d/%d unitaries' % (len(path_lens), len(goals)))
        print('Saving paths to `%s` | min/max/mean: %.2f/%.2f/%.2f' % (args.save_file, \
            min(path_lens), max(path_lens), sum(path_lens) / len(path_lens)))
        
        with open(args.save_file, 'wb') as f:
            pickle.dump({'goals': goals, 'paths': paths}, f)

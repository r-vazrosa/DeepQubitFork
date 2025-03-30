"""
Script that uses A* search to synthesize a
unitary matrix from an arbitrary gate set
"""
import pickle
from argparse import ArgumentParser
from typing import Dict, List, Optional
from deepxube.search.astar import AStar, get_path
from deepxube.nnet import nnet_utils
from environments.qcircuit import QCircuit, QGoal, QState, QAction
from utils.matrix_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--nnet_weights', type=str, required=True)
    parser.add_argument('--goals_file', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-2)
    args = parser.parse_args()

    # loading goal data
    data: Dict
    with open(args.goals_file, 'rb') as f:
        data = pickle.load(f)

    # environment setup
    env: QCircuit = QCircuit(num_qubits=data['num_qubits'], epsilon=args.epsilon)
    # goals: List[QGoal] = [QGoal(x) for x in data['unitaries']]
    goals = [QGoal(data['unitaries'][0])]
    start_states: List[QState] = [QState(tensor_product([I] * data['num_qubits'])) for _ in goals]
    weights: List[float] = [0.2] * len(start_states)

    # loading heuristic function
    device, devices, on_gpu = nnet_utils.get_device()
    heuristic_fn = nnet_utils.load_heuristic_fn(args.nnet_weights, device, on_gpu, env.get_v_nnet(), env)

    # setup A* search
    astar = AStar(env)
    astar.add_instances(start_states, goals, weights, heuristic_fn)

    # running search
    step: int = 0
    while np.any([not x.finished for x in astar.instances]) and step < args.max_steps:
        astar.step(heuristic_fn, args.batch_size, verbose=True)
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

    # saving paths
    print('Saving paths to `%s` | min/max/mean: %.2f/%.2f/%.2f' % (args.save_file, \
        min(path_lens), max(path_lens), sum(path_lens) / len(path_lens)))
    
    with open(args.save_file, 'wb') as f:
        pickle.dump({'goals': goals, 'paths': paths}, f)


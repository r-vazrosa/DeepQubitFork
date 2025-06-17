# DeepQubit

Compiling quantum circuits from arbitrary gate sets to approximate unitary operators using reinforcement learning and search
(based on [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf))

## Setup

1. Install project dependencies

    ```
    pip install -r requirements.txt
    ```

2. Run the project setup script every time you open a new shell

    ```
    source setup.sh
    ```

## Training

Use the train script, for example:

```
python scripts/train.py \
    --epsilon 0.01 \
    --num_qubits 2 \
    --nnet_dir tmp/2qubit \
    --step_max 100 \
    --batch_size 10000 \
    --itrs_per_update 1000 \
    --max_itrs 100000
```

The full list of command line options for `scripts/train.py` are listed below:

| Option | Usage |
| -- | -- |
| --num_qubits | Number of qubits to train the model for compilation on |
| --nnet_dir | Directory for storing the trained network weights |
| --epsilon | Float value of tolerance for accepting solution |
| --step_max | Number of steps to use for greedy policy test |
| --batch_size | Training update batch size |
| --itrs_per_update | Number of iterations per value iteration update |
| --max_itrs | Maxmimum training iterations |
| --greedy_update_step_max | How many steps to take during data generation in value iteration updates |
| --num_update_procs | Number of processes to run for updating in parallel |


## Heuristic search

To run A* heuristic search using a trained model run the search script; for example
```
python scripts/search.py target.txt --output circuit.qasm --epsilon 0.01
```

The full list of command line options for `scripts/search.py` are listed below:

| Option | Usage |
| -- | -- |
| --output / -o | Filename for saving circuit output |
| --epsilon / -e | Tolerance for solution acceptance |
| --max_steps / -m | Maximum steps allowed for A* search |
| --batch_size / -o | Batch size for A* search |
| --path_weight / -p | Path weight for A* search |
| --verbose / -v | Print all information for each A* update |
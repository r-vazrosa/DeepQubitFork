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
    --num_qubits 2 \
    --nnet_dir tmp/model \
    --step_max 100 \
    --batch_size 1000 \
    --itrs_per_update 1000 \
    --max_itrs 100000
```

## Heuristic search

In perform heuristic search using a model, first generate random goal states
```
python scripts/generate_goals.py \
    --num_qubits 2 \
    --num_goals 1000 \
    --pkl_file tmp/goals.pkl
```

Then run A* search to find paths to the goal states
```
python scripts/search.py \
    --nnet_dir tmp/model \
    --goals_file tmp/goals.pkl \
    --save_file tmp/paths.pkl
```
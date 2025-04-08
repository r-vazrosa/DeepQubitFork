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

## Random matrix generation

To generate 1000 random 1-qubit unitary operators run
```
python scripts/generate_goals.py \
    --num_qubits 1 \
    --num_goals 1000 \
    --pkl_file tmp/goals.pkl
```

## Heuristic search

To run A* heuristic search using a trained model run
```
python scripts/search.py \
    --epsilon 0.03 \
    --nnet_dir tmp/model \
    --goals_file tmp/goals.pkl \
    --save_file tmp/paths.pkl
```
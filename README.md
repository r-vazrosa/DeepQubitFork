# DeepQube

Compiling quantum circuits from arbitrary gate sets to approximate unitary operators using reinforcement learning and search
(based on [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf))

## Setup

1. Make sure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed

2. Create and activate conda environment
   
    ```
    conda create --name deepqube python=3.12
    conda activate deepqube
    ```

3. Install the project dependencies

    ```
    pip install -r requirements.txt
    ```

    Note: when installing dependencies on the UofSC research cluster, instead run
    ```
    TMPDIR=/work/`whoami`/pip_cache pip install --cache-dir /work/`whoami`/pip_cache -r requirements.txt
    ```

4. Run the project setup script

    ```
    source setup.sh
    ```

## Training

Use the train script, for example:

```
python scripts/train.py --num_qubits 2 --nnet_dir tmp/qcircuit2
```

## Evaluating

1. In order to evaluate a model, first generate random goal states
    ```
    python scripts/generate_goals.py --num_qubits 2 --num_goals 1000 --save_file tmp/goals.pkl
    ```

2. Then run A* search to find paths to the goal states
   ```
   python scripts/heur_search.py --nnet_weights tmp/qcircuit2/current.pt --goals_file tmp/goals.pkl --save_file tmp/paths.pkl
   ```
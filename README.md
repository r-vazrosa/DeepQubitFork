# DeepQube

Compiling quantum circuits from arbitrary gate sets to approximate unitary operators using reinforcement learning and search
(based on [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf))

## Setup

1. Make sure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed

2. Create a new conda environment using
   
    ```
    conda create --name deepqube python=3.12
    ```

3. Install the project dependencies

    ```
    conda install --channel pytorch pytorch
    pip install deepxube
    ```

## Training

To train a model for 2 qubit compilation, run

```
python scripts/train.py --num_qubits 2 --nnet_dir ./tmp/qcircuit2
```
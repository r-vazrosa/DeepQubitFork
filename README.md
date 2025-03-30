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

To train a model for 2 qubit compilation, run

```
python scripts/train.py --num_qubits 2 --nnet_dir tmp/qcircuit2
```

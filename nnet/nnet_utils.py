import yaml
from typing import Dict


def load_nnet_config(filename: str) -> Dict:
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
        return data
    

def save_nnet_config(config: Dict, filename: str):
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
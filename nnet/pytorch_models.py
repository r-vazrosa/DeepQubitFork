import torch
from torch import nn
from typing import List
from deepxube.environments.environment_abstract import HeurFnNNet
from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel


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
        x: torch.Tensor = states_goals_l[0].float()
        x = self.fc_input(x)
        x = self.resnet(x)
        x = self.fully_connected(x)
        return x
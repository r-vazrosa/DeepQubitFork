import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List
from deepxube.environments.environment_abstract import HeurFnNNet
from deepxube.nnet.pytorch_models import ResnetModel, FullyConnectedModel
    
    
def dephase(x: torch.Tensor) -> torch.Tensor:
    n, c, d = x.shape
    assert c == 2
    # Compute mean of each batch element
    x_r, x_i = x[:, 0], x[:, 1]
    m_r, m_i = x_r.mean(dim=-1), x_i.mean(dim=-1)

    ang = torch.atan2(m_i, m_r + (m_r == 0) * 1e-6)
    ang_conj = -ang
    ang_r, ang_i = torch.cos(ang_conj).unsqueeze(-1), torch.sin(ang_conj).unsqueeze(-1)

    y_r = x_r * ang_r - x_i * ang_i
    y_i = x_r * ang_i + x_i * ang_r

    y = torch.stack([y_r, y_i], dim=1)
    assert y.shape == (n, c, d)
    return y


class NeRFEmbedding(nn.Module):
    """Does NeRF embeddings for input Tensors."""
    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L
        self._emb_vec = torch.tensor(
            [torch.pi * 2 ** i for i in range(L)], requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = x.shape
        if self._emb_vec.device != x.device:
            self._emb_vec = self._emb_vec.to(x.device)
        l = self.L
        x = repeat(x, 'b d-> b d l', l=l)
        _sin = (x * self._emb_vec).sin()
        _cos = (x * self._emb_vec).cos()
        interleaved = torch.stack([_sin, _cos], dim=-1).view(b, d * l * 2)
        return interleaved


class ResnetModel(HeurFnNNet):
    def __init__(self, encoding_dim: int, state_dim: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super(ResnetModel, self).__init__(nnet_type='V')
        self.encoding_dim: int = encoding_dim
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # NeRF for input encoding
        self.nerf = NeRFEmbedding(L=encoding_dim)

        # first two hidden layers
        self.fc1 = nn.Linear(self.state_dim * encoding_dim * 2, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for _ in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_goals_l):
        # processing input
        x = states_goals_l[0]
        x = self.nerf(x).float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x
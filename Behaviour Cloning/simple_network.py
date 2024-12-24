import torch
import torch.nn as nn
import numpy as np


class LinearResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(LinearResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
    def forward(self, x):
        return x + self.main(x)

class SimpleNet(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_layer_dimension,
        max_episode_length=1600,
        device="cpu",
    ):
        super(SimpleNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dimension = hidden_layer_dimension
        self.max_episode_length = max_episode_length
        

        # learnable episode timestep embedding
        self.episode_timestep_embedding = nn.Embedding(self.max_episode_length, self.hidden_layer_dimension)

        # state embedding
        self.state_embedding = nn.Linear(self.state_dim, self.hidden_layer_dimension)

        # main linear layer
        self.main_model = nn.Sequential(
            LinearResidualBlock(self.hidden_layer_dimension, self.hidden_layer_dimension),
            LinearResidualBlock(self.hidden_layer_dimension, self.hidden_layer_dimension),
            nn.Linear(self.hidden_layer_dimension, self.action_dim),
            nn.Tanh()
        )

        self.device = device
        self.to(self.device)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state, timestep):
        return self.main_model(self.state_embedding(state) + self.episode_timestep_embedding(timestep))


        
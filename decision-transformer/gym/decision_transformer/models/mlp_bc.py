import numpy as np
import torch
import torch.nn as nn

from decision_transformer.models.model import TrajectoryModel


class MLPBCModel(TrajectoryModel):
    """
    Enhanced MLP that predicts next action a from past states s.
    This version has:
    - More hidden layers (increase n_layer value)
    - GELU activation instead of ReLU
    - Layer normalization after each hidden layer
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        # input layer
        layers = [nn.Linear(max_length * self.state_dim, hidden_size)]
        
        # more layers than before: after each layer, apply GELU, dropout, layer norm, then another linear
        for i in range(n_layer - 1):
            layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size)
            ])
        
        # final layers: activation, dropout, linear to output, Tanh to bound actions
        layers.extend([
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        # concatenate last `max_length` states
        states = states[:, -self.max_length:].reshape(states.shape[0], -1)
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length - states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0, -1]
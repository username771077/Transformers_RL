import numpy as np
import torch
import torch.nn as nn
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):
    """
    Enhanced Decision Transformer:
    - Increased default GPT2 layers and heads for more capacity (if not provided in kwargs).
    - Added dropout after embeddings for regularization.
    - Added a final LayerNorm after transformer output for more stable training.
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        if 'n_layer' not in kwargs:
            kwargs['n_layer'] = 6  # increase from default to get a deeper model
        if 'n_head' not in kwargs:
            kwargs['n_head'] = 8   # more attention heads than the default GPT-2 small

        config = transformers.GPT2Config(
            vocab_size=1,  # dummy
            n_embd=hidden_size,
            **kwargs
        )

        # GPT2Model that doesn't have default positional embeddings (handled manually)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        # Layer norm and dropout after embeddings to improve stability and regularize
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.1)  # add dropout to embeddings

        # Final layer normalization before predictions for more stable training
        self.final_ln = nn.LayerNorm(hidden_size)

        # Prediction heads
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack inputs: (R, s, a)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        # Apply layer norm and dropout on embeddings
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_inputs = self.dropout(stacked_inputs)

        # Adjust attention mask for stacked inputs
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # Pass through the transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # Reshape output back
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Apply final layer norm before predictions
        x = self.final_ln(x)

        # Predictions
        return_preds = self.predict_return(x[:, 2])  # next return
        state_preds = self.predict_state(x[:, 2])    # next state
        action_preds = self.predict_action(x[:, 1])  # next action

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Reshape input to single batch
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            # truncate sequences to max_length
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # Create and apply attention_mask
            seq_len = states.shape[1]
            attention_mask = torch.cat([torch.zeros(self.max_length - seq_len), torch.ones(seq_len)])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).unsqueeze(0)

            # Pad sequences to max_length
            if seq_len < self.max_length:
                pad_len = self.max_length - seq_len
                states = torch.cat([
                    torch.zeros((1, pad_len, self.state_dim), device=states.device), states
                ], dim=1).float()
                actions = torch.cat([
                    torch.zeros((1, pad_len, self.act_dim), device=actions.device), actions
                ], dim=1).float()
                returns_to_go = torch.cat([
                    torch.zeros((1, pad_len, 1), device=returns_to_go.device), returns_to_go
                ], dim=1).float()
                timesteps = torch.cat([
                    torch.zeros((1, pad_len), device=timesteps.device), timesteps
                ], dim=1).long()
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds[0, -1]
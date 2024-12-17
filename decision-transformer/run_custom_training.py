import torch
import torch.nn as nn
import numpy as np
import h5py
import os

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.trainer import Trainer
from decision_transformer.training.losses import sequence_loss

# Define the device: use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StateEncoder(nn.Module):
    """
    A CNN-based encoder to transform image observations into feature vectors.
    """
    def __init__(self, input_channels=3, encoded_dim=512):
        super(StateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: (32, 51, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # Output: (64, 24, 18)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # Output: (64, 22, 16)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, encoded_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # shape: (batch_size * T, encoded_dim)

class DecisionTransformerWithEncoder(nn.Module):
    """
    Combines the StateEncoder and DecisionTransformer into a single model.
    """
    def __init__(self, state_encoder, decision_transformer):
        super(DecisionTransformerWithEncoder, self).__init__()
        self.state_encoder = state_encoder
        self.decision_transformer = decision_transformer

    def forward(self, states, actions, rewards, attention_mask=None, returns_to_go=None):
        """
        Forward pass through the StateEncoder and DecisionTransformer.

        Args:
            states (torch.Tensor): Image observations, shape (batch_size, T, C, H, W)
            actions (torch.Tensor): Actions, shape (batch_size, T, act_dim)
            rewards (torch.Tensor): Rewards, shape (batch_size, T)
            attention_mask (torch.Tensor, optional): Attention masks, shape (batch_size, T)
            returns_to_go (torch.Tensor, optional): Returns-to-go, shape (batch_size, T)

        Returns:
            torch.Tensor: Predicted actions, shape (batch_size, T, act_dim)
        """
        batch_size, T, C, H, W = states.shape
        print(f"Forward pass: batch_size={batch_size}, T={T}, C={C}, H={H}, W={W}")

        # Reshape to (batch_size * T, C, H, W) for the encoder
        states = states.view(batch_size * T, C, H, W)
        print(f"Reshaped states for encoding: {states.shape}")

        # Encode states
        encoded_states = self.state_encoder(states)  # shape: (batch_size * T, encoded_dim)
        print(f"Encoded states shape: {encoded_states.shape}")

        # Reshape back to (batch_size, T, encoded_dim)
        encoded_states = encoded_states.view(batch_size, T, -1)
        print(f"Reshaped encoded states: {encoded_states.shape}")

        # Forward pass through DecisionTransformer
        output = self.decision_transformer(
            states=encoded_states,
            actions=actions,
            rewards=rewards,
            attention_mask=attention_mask,
            returns_to_go=returns_to_go
        )
        print(f"DecisionTransformer output shape: {output.shape}")

        return output
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """
        Generate actions given the current state and past context.

        Args:
            states (torch.Tensor): Image observations, shape (batch_size, T, C, H, W)
            actions (torch.Tensor): Past actions, shape (batch_size, T, act_dim)
            rewards (torch.Tensor): Past rewards, shape (batch_size, T)
            returns_to_go (torch.Tensor): Desired returns, shape (batch_size, T)
            timesteps (torch.Tensor): Timesteps, shape (batch_size, T)

        Returns:
            torch.Tensor: Generated actions, shape (batch_size, act_dim)
        """
        batch_size, T, C, H, W = states.shape
        print(f"Get_action: batch_size={batch_size}, T={T}, C={C}, H={H}, W={W}")

        # Reshape to (batch_size * T, C, H, W) for the encoder
        states = states.view(batch_size * T, C, H, W)
        print(f"Reshaped states for encoding (get_action): {states.shape}")

        # Encode states
        encoded_states = self.state_encoder(states)  # shape: (batch_size * T, encoded_dim)
        print(f"Encoded states shape (get_action): {encoded_states.shape}")

        # Reshape back to (batch_size, T, encoded_dim)
        encoded_states = encoded_states.view(batch_size, T, -1)
        print(f"Reshaped encoded states (get_action): {encoded_states.shape}")

        # Forward pass through DecisionTransformer
        output = self.decision_transformer.get_action(
            states=encoded_states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            **kwargs
        )
        print(f"DecisionTransformer get_action output shape: {output.shape}")

        return output

class HDF5Dataset:
    """
    A class to handle HDF5 data loading efficiently by keeping the file open
    and providing indexed access to the datasets.
    """
    def __init__(self, file_path):
        """
        Initializes the HDF5Dataset by opening the HDF5 file and setting up references
        to the required datasets without loading them into memory.

        Args:
            file_path (str): Path to the HDF5 file.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"HDF5 file not found at path: {file_path}")
        
        print(f"Opening HDF5 file at path: {file_path}")
        self.file = h5py.File(file_path, 'r')
        
        # Inspect and print dataset shapes
        print("\nInspecting HDF5 datasets...")
        for name in self.file.keys():
            data_shape = self.file[name].shape
            print(f"Dataset '{name}' shape: {data_shape}")
        
        self.actions = self.file['actions']          # Expected shape: (N, T, act_dim) or (N, T)
        self.observations = self.file['observations']# Expected shape: (N, T, 210, 160, 3)
        self.rewards = self.file['rewards']          # Expected shape: (N, T)
        self.terminals = self.file['terminals']      # Expected shape: (N, T)
        self.timeouts = self.file['timeouts']        # Expected shape: (N, T)
        self.N = self.actions.shape[0]               # Number of episodes

    def __len__(self):
        """Returns the number of episodes in the dataset."""
        return self.N

    def get_batch(self, indices, T):
        """
        Retrieves a batch of data for the given indices.

        Args:
            indices (array-like): Indices of the episodes to retrieve.
            T (int): Sequence length.

        Returns:
            Tuple of numpy arrays: states, actions, rewards, dones, returns, attention_mask
        """
        print(f"\nRetrieving batch for indices: {indices}")
        # Retrieve data for the given indices
        obs_batch = self.observations[indices]         # shape (batch_size, T, 210, 160, 3)
        actions_batch = self.actions[indices]          # shape (batch_size, T, act_dim) or (batch_size, T)
        rewards_batch = self.rewards[indices]          # shape (batch_size, T)
        terminals_batch = self.terminals[indices]      # shape (batch_size, T)
        timeouts_batch = self.timeouts[indices]        # shape (batch_size, T)

        # Debug: Print shapes of retrieved batches
        print("\nRetrieved batch shapes:")
        print(f"Observations batch shape: {obs_batch.shape}")
        print(f"Actions batch shape: {actions_batch.shape}")
        print(f"Rewards batch shape: {rewards_batch.shape}")
        print(f"Terminals batch shape: {terminals_batch.shape}")
        print(f"Timeouts batch shape: {timeouts_batch.shape}")

        # Preprocess the batch to compute returns and reshape data
        try:
            states, actions, rewards, dones, returns, attention_mask = self.preprocess_batch(
                obs_batch, actions_batch, rewards_batch, terminals_batch, timeouts_batch, T
            )
        except ValueError as e:
            print(f"Error during preprocessing: {e}")
            raise

        return states, actions, rewards, dones, returns, attention_mask

    def preprocess_batch(self, obs_batch, actions_batch, rewards_batch, terminals_batch, timeouts_batch, T):
        """
        Preprocesses a batch of data by computing returns and creating attention masks.

        Args:
            obs_batch (np.array): Observations, shape (batch_size, T, 210, 160, 3)
            actions_batch (np.array): Actions, shape (batch_size, T, act_dim) or (batch_size, T)
            rewards_batch (np.array): Rewards, shape (batch_size, T)
            terminals_batch (np.array): Terminals, shape (batch_size, T)
            timeouts_batch (np.array): Timeouts, shape (batch_size, T)
            T (int): Sequence length

        Returns:
            Tuple of numpy arrays: states, actions, rewards, dones, returns, attention_mask
        """
        batch_size = obs_batch.shape[0]
        act_dim = 1  # Default to 1, adjust if actions are multidimensional

        print("\nHandling actions shape...")
        if actions_batch.ndim == 2:
            # Expected shape: (batch_size, T)
            print("Actions are 2D. Reshaping to add act_dim=1.")
            actions_batch = actions_batch.reshape(batch_size, T, act_dim)  # shape (batch_size, T, act_dim)
        elif actions_batch.ndim == 3:
            # Expected shape: (batch_size, T, act_dim)
            act_dim = actions_batch.shape[2]
            print(f"Actions are 3D with act_dim={act_dim}.")
        else:
            print(f"Unsupported actions shape: {actions_batch.shape}")
            raise ValueError("Unsupported actions shape.")

        # Debug: Print actions_batch shape after handling
        print(f"Actions batch shape after handling: {actions_batch.shape}")

        # Keep observations as images (batch_size, T, 210, 160, 3)
        states = obs_batch  # shape (batch_size, T, 210, 160, 3)
        print(f"States shape (images): {states.shape}")

        # Compute returns-to-go
        print("\nComputing returns-to-go...")
        returns = np.zeros_like(rewards_batch, dtype=np.float32)
        for i in range(batch_size):
            G = 0
            for t in reversed(range(T)):
                G += rewards_batch[i, t]
                returns[i, t] = G
        print("Returns-to-go computed.")

        # Create attention mask (all ones for full-length sequences)
        attention_mask = np.ones((batch_size, T), dtype=np.float32)
        print(f"Attention mask shape: {attention_mask.shape}")

        # Dones are equivalent to terminals in this context
        dones = terminals_batch.astype(np.float32)
        print(f"Dones shape: {dones.shape}")

        return states, actions_batch, rewards_batch.astype(np.float32), dones, returns, attention_mask

    def close(self):
        """Closes the HDF5 file."""
        print("\nClosing HDF5 file.")
        self.file.close()

def make_get_batch_function(hdf5_dataset, T):
    """
    Creates a get_batch function that retrieves batches from the HDF5Dataset.

    Args:
        hdf5_dataset (HDF5Dataset): The dataset object.
        T (int): Sequence length.

    Returns:
        Function: A function that takes batch_size and returns a batch of data.
    """
    def get_batch(batch_size):
        # Randomly sample batch_size indices
        indices = np.random.randint(0, len(hdf5_dataset), size=batch_size)
        print(f"\nSampling batch indices: {indices}")
        # Retrieve the batch data
        states, actions, rewards, dones, returns, attention_mask = hdf5_dataset.get_batch(indices, T)
        
        # Convert to torch tensors and move to device
        print("\nConverting numpy arrays to torch tensors and moving to device...")
        try:
            s = torch.from_numpy(states).float().to(device)                     # shape (batch_size, T, C, H, W)
            a = torch.from_numpy(actions).float().to(device)                    # shape (batch_size, T, act_dim)
            r = torch.from_numpy(rewards).float().to(device)                    # shape (batch_size, T)
            d = torch.from_numpy(dones).float().to(device)                      # shape (batch_size, T)
            ret = torch.from_numpy(returns).float().to(device)                  # shape (batch_size, T)
            attn = torch.from_numpy(attention_mask).float().to(device)          # shape (batch_size, T)
        except Exception as e:
            print(f"Error during tensor conversion: {e}")
            raise

        # Debug: Print tensor shapes
        print(f"Tensor shapes:")
        print(f"States tensor shape: {s.shape}")
        print(f"Actions tensor shape: {a.shape}")
        print(f"Rewards tensor shape: {r.shape}")
        print(f"Dones tensor shape: {d.shape}")
        print(f"Returns tensor shape: {ret.shape}")
        print(f"Attention mask tensor shape: {attn.shape}")

        return s, a, r, d, attn, ret

    return get_batch

def main():
    # Path to your HDF5 file
    file_path = '/home/ivanlyubimtsev/Downloads/Galaxian-v55.h5'  # Updated path

    # Initialize the HDF5 dataset
    hdf5_dataset = HDF5Dataset(file_path)

    # Choose the sequence length T (must match your data segmentation)
    T = 50  # Adjust based on your data

    # Create the get_batch function
    batch_size = 64
    get_batch = make_get_batch_function(hdf5_dataset, T)

    # Extract dimensions from a sample batch to configure the model
    print("\nExtracting dimensions from a sample batch...")
    sample_indices = [0]
    try:
        sample_states, sample_actions, _, _, _, _ = hdf5_dataset.get_batch(sample_indices, T)
    except ValueError as e:
        print(f"Error while extracting sample batch: {e}")
        hdf5_dataset.close()
        return

    state_dim = 512  # Encoded dimension from StateEncoder
    act_dim = sample_actions.shape[-1]    # 1 or as per your action space
    print(f"\nConfigured state_dim: {state_dim}, act_dim: {act_dim}")

    hidden_size = 128
    max_length = T
    max_ep_len = 4096  

    # Initialize the State Encoder
    state_encoder = StateEncoder(input_channels=3, encoded_dim=state_dim).to(device)
    print(f"Initialized StateEncoder with encoded_dim={state_dim}.")

    # Initialize the Decision Transformer
    decision_transformer = DecisionTransformer(
        state_dim=state_dim,  # Using encoded_dim from StateEncoder
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
        max_ep_len=max_ep_len,
        action_tanh=True,
    ).to(device)
    print("Initialized DecisionTransformer.")

    # Combine the encoder and transformer into one model
    model = DecisionTransformerWithEncoder(state_encoder, decision_transformer).to(device)
    print("Combined StateEncoder and DecisionTransformer into DecisionTransformerWithEncoder.")

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = sequence_loss
    print("Initialized optimizer and loss function.")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        loss_fn=loss_fn,
        # scheduler=None,  
        # eval_fns=None,   
    )
    print("Initialized Trainer.")

    # Training loop
    num_iterations = 5 
    steps_per_iteration = 100

    for i in range(num_iterations):
        print(f"\n=== Starting iteration {i+1}/{num_iterations} ===")
        try:
            logs = trainer.train_iteration(num_steps=steps_per_iteration, iter_num=i, print_logs=True)
            print(f"=== Iteration {i+1} completed. Logs: {logs} ===")
        except Exception as e:
            print(f"Error during training iteration {i+1}: {e}")
            break

    # Close the HDF5 dataset after training
    hdf5_dataset.close()
    print("Training completed and HDF5 dataset closed.")

if __name__ == "__main__":
    main()

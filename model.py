"""ECG CNN model for federated learning with fog computing."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGModel(nn.Module):
    """Simple 1D-CNN for binary ECG classification (normal vs. anomaly).

    This model is designed for federated learning scenarios where:
    - Input: ECG time series data (140 time points)
    - Output: Binary classification logits
    - Architecture: Two conv-pool blocks + fully connected layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 140,
    ) -> None:
        """Initialize the ECG CNN model.

        Args:
            in_channels: Number of input channels (default: 1 for ECG)
            seq_len: Length of input sequence (default: 140 for ECG5000)
        """
        super().__init__()
        # Two conv → pool blocks
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5)  # → (16, 136)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)  # → (32, 132)
        self.pool = nn.MaxPool1d(2)  # halves the length

        # Compute flatten size after conv/pool layers
        length_after = self._get_length_after(seq_len)
        self.fc1 = nn.Linear(32 * length_after, 64)
        self.fc2 = nn.Linear(64, 1)  # single-logit output for BCEWithLogits

    def _get_length_after(self, L: int) -> int:
        """Calculate sequence length after conv and pooling operations.

        Args:
            L: Input sequence length

        Returns:
            Final sequence length after conv/pool operations
        """
        # conv1: L-4 → pool: (L-4)//2
        L = (L - 4) // 2
        # conv2: L-4 → pool: (L-4)//2
        L = (L - 4) // 2
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Raw logits of shape (batch_size, 1)
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract model weights as a list of NumPy arrays.

    This function is used in federated learning to extract model parameters
    for transmission to other nodes.

    Args:
        model: PyTorch model to extract parameters from

    Returns:
        List of NumPy arrays containing model parameters
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[torch.Tensor]) -> None:
    """Load parameters (list of NumPy arrays) into the model.

    This function is used in federated learning to update model parameters
    with weights received from other nodes.

    Args:
        model: PyTorch model to update
        parameters: List of NumPy arrays containing new parameters
    """
    state_dict = model.state_dict()
    for (key, _), array in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(array)
    model.load_state_dict(state_dict)

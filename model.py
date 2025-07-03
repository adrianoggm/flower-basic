import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGModel(nn.Module):
    """Simple 1D‐CNN for binary ECG classification (normal vs. anomaly)."""
    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 140,
    ):
        super().__init__()
        # Two conv → pool blocks
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5)  # → (16, 136)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)           # → (32, 132)
        self.pool  = nn.MaxPool1d(2)                            # halves the length

        # Compute flatten size after conv/pool layers
        length_after = self._get_length_after(seq_len)
        self.fc1 = nn.Linear(32 * length_after, 64)
        self.fc2 = nn.Linear(64, 1)  # ← single‐logit output for BCEWithLogits

    def _get_length_after(self, L: int) -> int:
        # conv1: L-4 → pool: (L-4)//2
        L = (L - 4) // 2
        # conv2: L-4 → pool: (L-4)//2
        L = (L - 4) // 2
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, seq_len)
        returns: (batch_size, 1) raw logits
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_parameters(model: nn.Module) -> list:
    """
    Returns model weights as a list of NumPy arrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    """
    Loads parameters (list of NumPy arrays) into the model.
    """
    state_dict = model.state_dict()
    for (key, _), array in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(array)
    model.load_state_dict(state_dict)

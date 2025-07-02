# =============================================================================
# client.py
# =============================================================================
import flwr as fl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import Net, get_weights, set_weights
from utils import load_data

# Flower client implementing NumPyClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data, model):
        self.cid = cid
        self.model = model
        X_train, y_train, X_test, y_test = data[cid]
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=32,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=32,
        )

    def get_parameters(self):
        return get_weights(self.model)

    def fit(self, parameters, config):
        # Set model parameters
        set_weights(self.model, parameters)
        # Train locally
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for epoch in range(1):  # Single epoch for demo
            for X, y in self.train_loader:
                optimizer.zero_grad()
                logits = self.model(X)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
        # Return updated weights and training size
        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Set model parameters
        set_weights(self.model, parameters)
        # Evaluate locally
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                logits = self.model(X)
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                preds = logits.argmax(axis=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        # Return loss, sample count, and metrics
        return loss, total, {"accuracy": correct / total}

if __name__ == "__main__":
    # Load data
    data = load_data(n_clients=2)
    # Create client model
    model = Net(num_features=20, num_classes=2)
    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(cid=0, data=data, model=model))
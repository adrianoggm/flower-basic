import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import load_ecg5000_openml
from model import ECGModel, get_parameters, set_parameters

class ECGClient(fl.client.NumPyClient):
    def __init__(self):
        X_train, X_test, y_train, y_test = load_ecg5000_openml()
        train_ds = TensorDataset(torch.from_numpy(X_train).unsqueeze(1).float(),
                                 torch.from_numpy(y_train))
        test_ds  = TensorDataset(torch.from_numpy(X_test).unsqueeze(1).float(),
                                 torch.from_numpy(y_test))
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=32)
        self.model = ECGModel()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        for X, y in self.train_loader:
            optimizer.zero_grad()
            logits = self.model(X).squeeze(1)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.BCEWithLogitsLoss()
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                logits = self.model(X).squeeze(1)
                total_loss += criterion(logits, y.float()).item() * X.size(0)
                preds = (torch.sigmoid(logits) > 0.5).int()
                correct += (preds == y).sum().item()
                total += X.size(0)
        return float(total_loss / total), total, {"accuracy": correct / total}

if __name__ == "__main__":
    # Reintentos infinitos cada 5s hasta conectar con éxito
    while True:
        try:
            fl.client.start_numpy_client(
                server_address="localhost:8080",
                client=ECGClient(),
            )
            break
        except ConnectionRefusedError:
            print("Servidor no disponible, reintentando en 5s…")
            time.sleep(5)

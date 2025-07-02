
# =============================================================================
# server.py
# =============================================================================
import flwr as fl
from model import Net, get_weights
from utils import load_data

if __name__ == "__main__":
    # Load sample data to infer shape (unused on server)
    _data = load_data(n_clients=2)
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        evaluate_fn=None,
        on_fit_config_fn=lambda rnd: {},
        on_evaluate_config_fn=lambda rnd: {},
    )
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
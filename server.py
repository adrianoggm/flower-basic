import flwr as fl
from typing import List, Tuple, Dict

def fit_config(rnd: int) -> Dict[str, int]:
    """Return training configuration dict for this round."""
    return {"lr": 1e-3, "local_epochs": 1}

def evaluate_config(rnd: int) -> Dict[str, int]:
    """Return evaluation configuration dict for this round."""
    return {}

def weighted_avg_accuracy(
    metrics: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """Aggregate client evaluation metrics (accuracy).  
    `metrics` is a list of (num_examples, {"accuracy": float}) tuples.
    """
    total_examples = sum(n for n, _ in metrics)
    # weighted sum of accuracies
    weighted_sum = sum(n * m["accuracy"] for n, m in metrics)
    return {"accuracy": weighted_sum / total_examples}


if __name__ == "__main__":
    # Definimos la estrategia FedAvg con un mínimo de 2 clientes
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # todos los clientes participarán
        fraction_evaluate=1.0,        # todos los clientes participarán en la evaluación
        min_fit_clients=2,            # esperar al menos 2 clientes para fit()
        min_evaluate_clients=2,       # esperar al menos 2 clientes para evaluate()
        min_available_clients=2,      # no empezar hasta tener 2 clientes “connectados”
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_avg_accuracy,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
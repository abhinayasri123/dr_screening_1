import flwr as fl
from typing import Dict, List, Optional, Tuple
from flwr.common import Metrics

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    return {}

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}
    
    weighted_acc = sum([num_examples * m.get("accuracy", 0) for num_examples, m in metrics]) / total_examples
    weighted_qwk = sum([num_examples * m.get("qwk", 0) for num_examples, m in metrics]) / total_examples
    
    return {"accuracy": weighted_acc, "qwk": weighted_qwk}

def create_strategy(config, initial_parameters):
    
    def on_fit_config_fn(server_round: int):
        return {
            "epochs": config.get("local_epochs", 1),
            "lr": config.get("lr", 1e-4),
            "mu": config.get("mu", 0.0) # For FedProx
        }
        
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.get("min_fit_clients", 2),
        min_evaluate_clients=config.get("min_evaluate_clients", 2),
        min_available_clients=config.get("min_available_clients", 2),
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters) if initial_parameters else None,
        on_fit_config_fn=on_fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    return strategy

def start_fl_server(config, strategy):
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.get("num_rounds", 5)),
        strategy=strategy
    )

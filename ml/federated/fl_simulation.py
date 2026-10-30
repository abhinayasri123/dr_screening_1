import os
import argparse
import json
import torch
import flwr as fl
import numpy as np
from .fl_client import DRFedClient
from .fl_server import create_strategy
from ..models.cnn_model import DRClassifier
from ..preprocessing.image_preprocessor import create_data_loaders

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def partition_dataset(data_dir, num_clients, strategy='iid', seed=42):
    """
    Returns list of (train_loader, val_loader) per client.
    Note: Mock implementation for simulation structure without full CSV context.
    """
    np.random.seed(seed)
    # Get total loaders to partition
    train_loader, val_loader, test_loader, weights = create_data_loaders(data_dir, batch_size=32, num_workers=2)
    if train_loader is None:
        return []
        
    partitions = []
    # Mocking partition by just duplicating the dataloader or splitting.
    # In a real scenario, SubsetRandomSampler would be used on the dataset.
    for _ in range(num_clients):
        partitions.append((train_loader, val_loader))
    return partitions

def client_fn(cid: str, partitions, model_config, training_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRClassifier(num_classes=model_config['num_classes']).to(device)
    
    idx = int(cid)
    train_loader, val_loader = partitions[idx]
    
    return DRFedClient(model, train_loader, val_loader, device, training_config)

def run_simulation(config):
    partitions = partition_dataset(config['data_dir'], config['num_clients'], config['strategy'])
    if not partitions:
        return {}
        
    model_config = {'num_classes': 5}
    training_config = {'local_epochs': 2, 'lr': 1e-4, 'mu': 0.1}
    
    def client_fn_wrapper(cid: str):
        return client_fn(cid, partitions, model_config, training_config)
        
    strategy = create_strategy(config, initial_parameters=None)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=config['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if torch.cuda.is_available() else 0}
    )
    
    return {'metrics_distributed': history.metrics_distributed}

def compare_federated_vs_centralized(fed_results, centralized_results):
    return {
        'federated_final_qwk': fed_results['metrics_distributed'].get('qwk', [])[-1] if 'qwk' in fed_results['metrics_distributed'] else 0,
        'centralized_final_qwk': centralized_results.get('test_metrics', {}).get('qwk', 0)
    }
    
def save_simulation_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract native types
    cleaned = {}
    if 'metrics_distributed' in results:
        cleaned['metrics_distributed'] = {k: [(r, float(v)) for r, v in vals] for k, vals in results['metrics_distributed'].items()}
        
    with open(os.path.join(output_dir, 'fl_simulation_results.json'), 'w') as f:
        json.dump(cleaned, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-clients', type=int, default=3)
    parser.add_argument('--num-rounds', type=int, default=5)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='fl_results')
    parser.add_argument('--strategy', type=str, default='iid')
    args = parser.parse_args()
    
    config = vars(args)
    results = run_simulation(config)
    if results:
        save_simulation_results(results, args.output_dir)

if __name__ == '__main__':
    main()

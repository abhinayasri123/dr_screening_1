#!/usr/bin/env python3
"""
Federated Learning Simulation Runner
======================================
Runs FL simulation using Flower (flwr), compares with centralized training,
and saves per-round metrics for analysis.

Usage:
    python scripts/run_fl_simulation.py
    python scripts/run_fl_simulation.py --rounds 10 --clients 3 --output-dir logs/federated
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args():
    p = argparse.ArgumentParser(description="Run Federated Learning simulation")
    p.add_argument("--config",      type=str, default="configs/federated_config.yaml")
    p.add_argument("--rounds",      type=int, default=None,
                   help="Override number of FL rounds from config")
    p.add_argument("--clients",     type=int, default=None,
                   help="Override number of simulated hospital clients")
    p.add_argument("--data-dir",    type=str, default="data/retinal")
    p.add_argument("--output-dir",  type=str, default="logs/federated")
    p.add_argument("--no-centralized", action="store_true",
                   help="Skip centralized baseline training")
    return p.parse_args()


def load_config(config_path: Path) -> dict:
    import yaml
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    print(f"⚠️  Config not found: {config_path}. Using defaults.")
    return {
        "simulation": {"num_clients": 3, "partition_strategy": "iid", "random_seed": 42},
        "server": {"num_rounds": 5, "strategy": "fedavg",
                   "min_fit_clients": 2, "min_evaluate_clients": 2,
                   "min_available_clients": 3},
        "client": {"local_epochs": 2, "batch_size": 16, "learning_rate": 1e-4},
        "comparison": {"run_centralized_baseline": True, "save_round_metrics": True},
    }


def main():
    args = parse_args()

    config_path = _project_root / args.config
    cfg = load_config(config_path)

    # Apply CLI overrides
    if args.rounds:
        cfg["server"]["num_rounds"] = args.rounds
    if args.clients:
        cfg["simulation"]["num_clients"] = args.clients

    data_dir   = _project_root / args.data_dir
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Federated Learning Simulation")
    print("=" * 60)
    print(f"  Clients      : {cfg['simulation']['num_clients']}")
    print(f"  FL rounds    : {cfg['server']['num_rounds']}")
    print(f"  Strategy     : {cfg['server']['strategy'].upper()}")
    print(f"  Partition    : {cfg['simulation']['partition_strategy'].upper()}")
    print(f"  Data dir     : {data_dir}")
    print(f"  Output dir   : {output_dir}")

    # Check data
    if not data_dir.exists() or not any(data_dir.rglob("*.jpg")) and not any(data_dir.rglob("*.png")):
        print("\n⚠️  No images found in", data_dir)
        print("   Run scripts/setup_dataset.py first to download and organise the dataset.")
        print("   Or run scripts/create_demo_data.py for a quick demo dataset.")
        sys.exit(1)

    # Run simulation
    from ml.federated.fl_simulation import run_simulation, save_simulation_results

    print("\nStarting FL simulation …")
    fl_results = run_simulation(cfg, str(data_dir))

    # Centralized baseline
    centralized_results = None
    if not args.no_centralized and cfg.get("comparison", {}).get("run_centralized_baseline", True):
        print("\nTraining centralized baseline for comparison …")
        try:
            from ml.federated.fl_simulation import train_centralized_baseline
            centralized_results = train_centralized_baseline(cfg, str(data_dir))
        except Exception as e:
            print(f"  Centralized training failed: {e}")
            centralized_results = None

    # Comparison
    comparison = None
    if fl_results and centralized_results:
        from ml.federated.fl_simulation import compare_federated_vs_centralized
        comparison = compare_federated_vs_centralized(fl_results, centralized_results)

    # Save results
    save_simulation_results(fl_results, centralized_results, comparison, str(output_dir))

    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    if fl_results:
        final_round = fl_results.get("rounds", [{}])[-1]
        print(f"  FL Final Accuracy : {final_round.get('accuracy', 'N/A'):.4f}")
        print(f"  FL Final QWK      : {final_round.get('qwk', 'N/A'):.4f}")
    if centralized_results:
        print(f"  Centralized Acc   : {centralized_results.get('accuracy', 'N/A'):.4f}")
        print(f"  Centralized QWK   : {centralized_results.get('qwk', 'N/A'):.4f}")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"    fl_results.json")
    print(f"    centralized_results.json")
    print(f"    fl_vs_centralized.json")


if __name__ == "__main__":
    main()

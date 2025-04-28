'''
Author: David Megli
Date: 2025-04-28
'''
import os
import yaml
import copy
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Definiamo i parametri dell'esperimento
depth_list = [2, 4, 8, 16, 32]
models = ['ParametrizedMLP', 'ResidualMLP']
# models = ['ResidualMLP']  # Solo per testare il codice
base_config_path = "configs/base_config.yaml"  # Config base comune
output_dir = "outputs/experiments"
results_file = os.path.join(output_dir, "results.csv")

os.makedirs(output_dir, exist_ok=True)

def run_training(model_name, depth, config_base, run_id):
    config = copy.deepcopy(config_base)

    config['model']['name'] = model_name

    if model_name == 'ParametrizedMLP':
        input_dim = config['model']['params']['input_dim']
        hidden_dim = config['model']['params']['hidden_dim']
        output_dim = config['model']['params'].get('output_dim', 10)  # es. MNIST
        # Costruzione di layer_sizes
        layer_sizes = [input_dim] + [hidden_dim] * depth + [output_dim]
        config['model']['params'] = {'layer_sizes': layer_sizes}
    
    elif model_name == 'ResidualMLP':
        config['model']['params']['num_blocks'] = depth

    config['trainer']['run_name'] = f"{model_name}_depth{depth}_{run_id}"
    config['output_dir'] = os.path.join(output_dir, config['trainer']['run_name'])

    temp_config_path = f"temp_config_{model_name.lower()}_{depth}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Training
    subprocess.run(["python", "train.py", "--config", temp_config_path])
    os.remove(temp_config_path)

    # Caricamento metriche salvate da train.py
    metrics_path = os.path.join(config['output_dir'], "metrics.yaml")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = yaml.safe_load(f)
        new_row = {
            "model": model_name,
            "depth": depth,
            "train_loss": metrics.get("train_loss", None),
            "train_accuracy": metrics.get("train_accuracy", None),
            "val_loss": metrics.get("val_loss", None),
            "val_accuracy": metrics.get("val_accuracy", None),
            "run_name": metrics.get("run_name", None),
            "parameters": metrics.get("parameters", None),
            "training_time": metrics.get("training_time", None),
        }
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(results_file, index=False)


def plot_results(results_path):
    df = pd.read_csv(results_path)

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        model_df = model_df.sort_values('depth')

        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(model_df['depth'], model_df['train_loss'], label='Train Loss')
        plt.plot(model_df['depth'], model_df['val_loss'], label='Val Loss')
        plt.title(f'{model_name} - Loss vs Depth')
        plt.xlabel('Depth')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(model_df['depth'], model_df['val_accuracy'], marker='o', label='Val Accuracy')  # AGGIORNATO QUI
        plt.title(f'{model_name} - Val Accuracy vs Depth')
        plt.xlabel('Depth')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_plots.png"))
        print(f"Plots saved for {model_name} in {os.path.join(output_dir, f'{model_name}_plots.png')}")
        plt.show()

if __name__ == "__main__":
    # Carichiamo la configurazione base
    with open(base_config_path, 'r') as f:
        config_base = yaml.safe_load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(results_file):
        pd.DataFrame(columns=[
            'model', 'depth', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
            'run_name', 'parameters', 'training_time'
        ]).to_csv(results_file, index=False)

    for model in models:
        for depth in depth_list:
            print(f"Starting experiment: {model} with depth {depth}")
            run_training(model, depth, config_base, run_id)
    
    plot_results(results_file)

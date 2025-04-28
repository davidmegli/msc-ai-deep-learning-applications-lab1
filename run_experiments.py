'''
Author: David Megli
Date: 2025-04-28
'''
import os
import yaml
import copy
import subprocess
from datetime import datetime

# Definiamo i parametri dell'esperimento
depth_list = [2, 4, 8, 16, 32]
models = ['ParametrizedMLP', 'ResidualMLP']
base_config_path = "configs/base_config.yaml"  # Config base comune
output_dir = "outputs/experiments"

os.makedirs(output_dir, exist_ok=True)

def run_training(model_name, depth, config_base, run_id):
    config = copy.deepcopy(config_base)

    config['model']['name'] = model_name

    if model_name == 'ParametrizedMLP':
        input_dim = config['model']['params']['input_dim']
        hidden_dim = config['model']['params']['hidden_dim']
        output_dim = config['model']['params'].get('output_dim', 10)  # es. MNIST
        # Costruiamo layer_sizes
        layer_sizes = [input_dim] + [hidden_dim] * depth + [output_dim]
        config['model']['params'] = {'layer_sizes': layer_sizes}
    
    elif model_name == 'ResidualMLP':
        config['model']['params']['num_blocks'] = depth

    config['trainer']['run_name'] = f"{model_name}_depth{depth}_{run_id}"
    config['output_dir'] = os.path.join(output_dir, config['trainer']['run_name'])

    temp_config_path = f"temp_config_{model_name.lower()}_{depth}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    subprocess.run(["python", "train.py", "--config", temp_config_path])
    os.remove(temp_config_path)

if __name__ == "__main__":
    # Carichiamo la configurazione base
    with open(base_config_path, 'r') as f:
        config_base = yaml.safe_load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in models:
        for depth in depth_list:
            print(f"Starting experiment: {model} with depth {depth}")
            run_training(model, depth, config_base, run_id)

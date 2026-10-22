import json
import os
import random
import argparse
from config import UZH_THESIS
parser = argparse.ArgumentParser()
parser.add_argument('--n_trials', type=int, default=15, help='Number of trials to generate')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

n_trials = args.n_trials
random.seed(args.seed)

configs_dir = os.path.join(UZH_THESIS, 'ArchConfigs')
os.makedirs(configs_dir, exist_ok=True)

search_space = {
    'n_blocks': [2, 3, 4],
    'base_channels': [8, 16, 32],
    'kernel_size': [3, 5],
    'dropout': [0.3, 0.5, 0.7],
    'activation': ['relu', 'gelu', 'leakyrelu'],
    'pool_type': ['max', 'avg'],
    'pool_size': [2, 3, 4],
    'fc_dims': [[32], [64], [32, 16]],
}


# Sample random configs (avoid duplicates)
seen = set()
trials = []
while len(trials) < n_trials:
    config = {k: random.choice(v) for k, v in search_space.items()}
    key = json.dumps(config, sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    trials.append(config)

# Save each as a JSON file
for i, config in enumerate(trials):
    config_name = f'arch_trial_{i:02d}'
    path = os.path.join(configs_dir, f'{config_name}.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'{config_name}: {config}')

print(f'\nGenerated {len(trials)} configs in {configs_dir}')
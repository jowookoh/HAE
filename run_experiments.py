import argparse
from train import train, load_config
from pathlib import Path

EXPERIMENT_SETS = {
    'cifar': [
        'configs/cifar_hypersphere_transformer.yaml',
        'configs/cifar_hypersphere_cnn.yaml',
        'configs/cifar_vae_transformer.yaml',
        'configs/cifar_vae_cnn.yaml',
    ],
    'quickdraw': [
        'configs/quickdraw_hypersphere_transformer.yaml',
        'configs/quickdraw_hypersphere_cnn.yaml',
        'configs/quickdraw_vae_transformer.yaml',
        'configs/quickdraw_vae_cnn.yaml',
    ],
    'all': [],  # populated below
}
EXPERIMENT_SETS['all'] = EXPERIMENT_SETS['cifar'] + EXPERIMENT_SETS['quickdraw']


def run_experiments(config_paths):
    total = len(config_paths)
    for i, config_path in enumerate(config_paths, 1):
        if not Path(config_path).exists():
            print(f"Skipping {config_path} (not found)")
            continue
        
        config = load_config(config_path)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i}/{total}: {config['name']}")
        print(f"Config: {config_path}")
        print(f"{'='*60}\n")
        
        train(config)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', '-s', type=str, default='quickdraw',
                        choices=list(EXPERIMENT_SETS.keys()),
                        help='Which experiment set to run')
    parser.add_argument('--configs', '-c', nargs='+', default=None,
                        help='Specific config files to run (overrides --set)')
    args = parser.parse_args()
    
    if args.configs:
        configs = args.configs
    else:
        configs = EXPERIMENT_SETS[args.set]
    
    run_experiments(configs)

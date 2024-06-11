'''
    NOT WORKING YET
'''

import yaml
import argparse

def update_config(config_path, updates):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the configuration with the provided updates
    for key, value in updates.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = yaml.safe_load(value)
    
    # Save the updated configuration
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)

if __name__=='__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Update YAML configuration file')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--params', nargs='+', required=True, help='Parameters to update in the configuration file, in the format key1.key2=value')

    args = parser.parse_args()

    updates = dict(param.split('=') for param in args.params)
    update_config(args.config, updates)

    # Load updated configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Run through the updated configuration
    for key, value in config.items():
        print(f'{key}')
        for key2, value2 in value.items():
            print(f'    {key2}: {value2}')
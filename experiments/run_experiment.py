import argparse
import yaml
import subprocess

def run(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running experiment: {config['experiment_name']}")
    subprocess.run(["python", "main.py", "--config", config_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    run(args.config)

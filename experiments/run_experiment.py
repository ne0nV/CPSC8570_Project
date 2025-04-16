"""
run_experiment.py

Script to run experiments with varying parameters.
Currently, it simply runs the main simulation.
"""

import subprocess

if __name__ == '__main__':
    print("Running experiments on enhanced federated learning defense mechanisms...")
    subprocess.run(["python", "main.py"])
    # Consider extending to loop over various parameters and log results.
    # For now, the main simulation is executed via main.py.

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

# Check backend
headless = matplotlib.get_backend() == "agg"

def plot_accuracy(log_path="logs/metrics.csv"):
    if not os.path.exists(log_path):
        print(f"No metrics file found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    if 'round' not in df.columns or 'accuracy' not in df.columns:
        print("Metrics file missing required columns: 'round' and 'accuracy'")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df['round'], df['accuracy'], marker='o', linestyle='-')
    plt.xlabel('Training Round')
    plt.ylabel('Validation Accuracy')
    plt.title('Global Model Accuracy Over Rounds')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure automatically
    save_path = log_path.replace("csv","png")
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

    # Show the plot if not headless
    if not headless:
        plt.show()

if __name__ == "__main__":
    plot_accuracy()

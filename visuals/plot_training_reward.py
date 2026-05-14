from matplotlib import pyplot as plt

from .train_run import TrainRun
from .load_runs import load_runs

def plot_adapted_rewards(runs: list[TrainRun], velocity_label: bool = True, flattenings: int = 10, save_fig: bool = True):
    for run in runs:
        df = run.get_returns(velocity_label=velocity_label, interpolate=True)
        for _ in range(flattenings):
            df = df.rolling(window=100).mean()
        
        title = f"Return after Training Adaptation - {run.selector}"
        labels = [f"{round(v, 2)}" for v in run.velocities] if velocity_label else df.columns + 1
        plt.plot(df, label=labels)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.grid()
        if save_fig:
            plt.savefig(f"plots/{title} - Seed {run.seed}", bbox_inches='tight')
        plt.show()
        plt.show()
        
if __name__ == "__main__":
    runs = load_runs(is_train=True)
    plot_adapted_rewards(runs)
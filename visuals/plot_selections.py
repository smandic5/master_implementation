from matplotlib import pyplot as plt
import numpy as np

from .train_run import TrainRun
from .load_runs import load_runs

def plot_selections(runs: list[TrainRun], velocity_label: bool = True, align_velocitys: bool = False, save_fig: bool = True):
    for run in runs:
        env_to_count = run.get_selection_counts(velocity_label=velocity_label)
        index = env_to_count.index.round(2) if align_velocitys else np.arange(len(env_to_count)) + 1
        width = 0.5 if align_velocitys else 1
        plt.bar(index, env_to_count.values, width=width)
        
        title = f"Selected Environments - {run.selector}"
        if not align_velocitys:
            pos = np.arange(20) + 1
            label = [f"{round(v, 2)}" for v in run.velocities] if velocity_label else pos
            plt.xticks(pos, label, rotation=45 if velocity_label else 0)
        plt.title(title)
        plt.ylabel("Number of selections")
        plt.xlabel("Environment velocity (units/s)" if velocity_label else "Environments")
        if save_fig:
            plt.savefig(f"plots/{title} - Seed {run.seed}.pdf", bbox_inches='tight', format="pdf")
        plt.show()

if __name__ == "__main__":
    runs = load_runs(is_train=True)
    plot_selections(runs)
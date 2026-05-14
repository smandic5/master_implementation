import matplotlib.pyplot as plt
import pandas as pd

from .task_selectors import get_selector_name
from .train_run import TrainRun

def extract_averages(
    runs: list[TrainRun],
    flattenings: int = 10,
    max_steps: int = 60000
) -> pd.DataFrame:
    avgs: list[pd.Series] = []
    for i in range(14):
        selector_runs = [run for run in runs if run.selector == get_selector_name(i)]
        mean_returns = [run.get_average_return(max_steps=max_steps) for run in selector_runs]
        for j, mean_return in enumerate(mean_returns):
            mean_return.name = j
        avg = pd.concat(mean_returns, axis=1, join="outer").mean(axis=1)
        avg.name = get_selector_name(i)
        avgs.append(avg)
    df: pd.DataFrame = pd.concat(avgs, axis=1, join="outer")
    for _ in range(flattenings):
        df = df.rolling(window=10).mean()
    return df

def plot_average_rewards(
    runs: list[TrainRun],
    flattenings: int = 10,
    max_steps: int = 60000,
    save_fig: bool = True
):
    df = extract_averages(runs, flattenings, max_steps)
    title = f"Average Adapted Returns"
    for i, column in enumerate(df.columns):
        color = plt.cm.tab20(i / 14)
        plt.plot(df[column], label=column, color=color)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if save_fig:
        plt.savefig(f"plots/{title}.pdf", bbox_inches='tight', format="pdf")
    plt.show()
    plt.show()
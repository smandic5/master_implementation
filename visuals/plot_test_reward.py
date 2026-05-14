import matplotlib.pyplot as plt
import pandas as pd

from .task_selectors import get_selector_name
from .test_run import TestRun

def extract_test_rewards(runs: list[TestRun], flattenings=10, max_steps=500) -> pd.DataFrame:
    avgs: list[pd.Series] = []
    for i in range(14):
        selector_runs = [run for run in runs if run.selector == get_selector_name(i)]
        returns = [run.get_return(max_steps=max_steps) for run in selector_runs]
        for j, r in enumerate(returns):
            r.name = j
        avg = pd.concat(returns, axis=1, join="outer").mean(axis=1)
        avg.name = get_selector_name(i)
        avgs.append(avg)
    df: pd.DataFrame = pd.concat(avgs, axis=1, join="outer")
    for _ in range(flattenings):
        df = df.rolling(window=10).mean()
    return df

def plot_test_reward(runs: list[TestRun], flattenings=10, max_steps=500, save_fig=True):
    dfs: list[pd.DataFrame] = []
    for e in range(3):
        env_runs = [run for run in runs if run.test_env == e]
        df = extract_test_rewards(env_runs, flattenings=flattenings, max_steps=max_steps)
        dfs.append(df)
    avgs: list[pd.Series] = []
    for column in dfs[0].columns:
        avg = pd.concat([df[column] for df in dfs], axis=1, join="outer").mean(axis=1)
        avg.name = column
        avgs.append(avg)
    df: pd.DataFrame = pd.concat(avgs, axis=1, join="outer")
    dfs.append(df)
        
    for e, df in enumerate(dfs):
        vel = f"Velocity  {[1.0, 4.0, 8.0][e]}" if e < 3 else "Average"
        title = f"Return after Test Adaptation - {vel}"
        for i, column in enumerate(df.columns):
            color = plt.cm.tab20(i / 14)
            plt.plot(df[column], label=column, color=color)
        plt.title(title)
        plt.ylabel("Return")
        plt.xlabel("Episode")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if save_fig:
            plt.savefig(f"plots/{title}.pdf", bbox_inches='tight', format="pdf")
        plt.show()
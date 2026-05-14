import json
import numpy as np
import pandas as pd

from .task_selectors import get_selector_name

TARGET_STAT = "adapted_reward"

class TrainRun():
    def __init__(self, dir: str, filename: str, max_velocity: float = 10.0, num_envs: int = 20):
        filename_splits = filename.split("_")
        self.env_name = filename_splits[0]
        self.alg_name = filename_splits[1]
        self.seed = int(filename_splits[2])
        self.selector = get_selector_name(int(filename_splits[3]))
        self.id = filename_splits[4].split(".")[0]
        self.filename = filename
        
        np.random.seed(self.seed)
        self.velocities = np.random.uniform(0.0, max_velocity, num_envs).tolist()
        
        with open(f"{dir}{filename}", "r") as f:
            self.stats: dict = json.load(f)
            self.stats_loc = self.stats.pop("loc")

        env_rewards: list[pd.Series] = []
        for full_stat_name, values in self.stats.items():
            if TARGET_STAT not in full_stat_name:
                continue
            splits = str(full_stat_name).split(" ")
            env_index = int(splits[1])
            values_positions = self.stats_loc[full_stat_name]
            series = pd.Series(values, values_positions, name=env_index)
            env_rewards.append(series)
        self.adapted_reward: pd.DataFrame = pd.concat(env_rewards, axis=1, join="outer")
        self.adapted_reward.sort_index(inplace=True)
        self.adapted_reward = self.adapted_reward.reindex(sorted(self.adapted_reward.columns), axis=1)
        self.adapted_reward.columns = self.velocities
        self.adapted_reward = self.adapted_reward.reindex(sorted(self.adapted_reward.columns), axis=1)
        self.adapted_reward.columns = np.arange(len(self.velocities), dtype=int)
        self.velocities = sorted(self.velocities)
        
    def __str__(self):
        return f"{self.selector} with seed {self.seed} - {self.id}"
    
    def get_returns(self, velocity_label: bool = True, interpolate: bool = True, max_steps: int = 60000) -> pd.DataFrame:
        df = self.adapted_reward.copy()
        if velocity_label:
            df.columns = self.velocities
        if interpolate:
            df = df.interpolate('index')
        return df[:max_steps]
    
    def get_return(self, env: int, velocity_label: bool = True, interpolate: bool = True, max_steps: int = 60000) -> pd.Series:
        df = self.get_returns(velocity_label, interpolate, max_steps)
        return df[env]
    
    def get_average_return(self, max_steps: int = 60000) -> pd.Series:
        df = self.get_returns(interpolate=True, max_steps=max_steps)
        return df.mean(axis=1)
    
    def get_selection_counts(self, velocity_label: bool = True, max_steps: int = 60000) -> pd.Series:
        return self.get_returns(velocity_label, interpolate=False, max_steps=max_steps).count()

    
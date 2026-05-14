import json
import pandas as pd

from .task_selectors import get_selector_name, get_selector_old_name

TARGET_STAT = "episodic_return"

class TestRun():
    def __init__(self, dir: str, filename: str):
        filename_splits = filename.split("_")
        self.selector = [get_selector_name(i) for i in range(14) if filename_splits[2] in [get_selector_old_name(i), get_selector_name(i)]][0]
        self.seed = int(filename_splits[4])
        self.iteration = int(filename_splits[6])
        self.test_env = int(filename_splits[8].split(".")[0])
        self.velocity = [1.0, 4.0, 8.0][self.test_env - 1]
        self.filename = filename
        
        # load and sort stats
        with open(f"{dir}{filename}", "r") as f:
            self.stats: dict = json.load(f)
            self.stats_loc = self.stats.pop("loc")
        self.rewards = pd.Series(self.stats[TARGET_STAT])
        
    def __str__(self):
        return f"{self.selector} with seed {self.seed} on {self.velocity}"
    
    def get_return(self, max_steps: int = 1000) -> pd.Series:
        return self.rewards.loc[:max_steps]
import os

from .train_run import TrainRun
from .test_run import TestRun

RESULT_DIR = "experiments/"
MODEL_DIR = "runs/"
STAT_DIR = "stats/"

stat_directory = RESULT_DIR + STAT_DIR

def load_runs(is_train: bool) -> list[TrainRun | TestRun]:
    all_run_names = os.listdir(stat_directory)
    needed_runs = [f for f in all_run_names if (f.split("_")[0] != "Eval") == is_train]
    if is_train:
        runs = [TrainRun(stat_directory, name) for name in needed_runs]
    else:
        runs = [TestRun(stat_directory, name) for name in needed_runs]
    runs.sort(key=lambda r: f"{r.selector}-{r.seed}")
    return runs

if __name__ == "__main__":
    runs = load_runs(is_train=True)
    for r in runs:
        print(r)
    
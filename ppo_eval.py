import os
import sys
import gymnasium as gym
from src.checkpoint import load_model, checkpoint
import torch.optim as optim
from src.args import init_args, Args
from src.envs.env_sets import init_test_envs_set
from src.logger import init_logger
from src.ppo.storage import DataHolder
from src.seeds import init_seeds
from src.ppo.train_ppo import train_ppo

from src.checkpoint import RESULT_DIR, MODEL_DIR, STAT_DIR
from src.task_selectors.factory import get_selector_name

stat_directory = RESULT_DIR + STAT_DIR
model_directory = RESULT_DIR + MODEL_DIR

class Run():
    def __init__(self, filename: str):
        filename_splits = filename.split("_")
        self.env_name = filename_splits[0]
        self.alg_name = filename_splits[1]
        self.seed = int(filename_splits[2])
        self.selector = get_selector_name(int(filename_splits[3]))
        self.id = filename_splits[4]
        self.filename = filename
        
        self.model_ids = [int(f.split(".")[0]) for f in os.listdir(f"{model_directory}{filename}/")]
        self.model_ids.sort()
        
    def __str__(self):
        return f"{self.selector} with seed {self.seed} - {self.id}"
        
    def get_agent(self, envs: gym.vector.VectorEnv, iteration: int):
        return load_model(self.filename, iteration, envs, "cpu")
    

def eval_model_env(
    run: Run,
    model_iteration: int,
    seed: int,
    args: Args,
    device,
    envs_test_set: list[gym.vector.SyncVectorEnv],
    env_i: int,
):
    run_name = f"Eval_Selector_{run.selector}_Seed_{run.seed}_Iteration_{model_iteration}_envs_{env_i}"
    print(f"Started {run_name}")
    logger = init_logger(args, run_name)
    init_seeds(seed, args.torch_deterministic)
    envs = envs_test_set[env_i]

    #agent = run.get_agent(envs, model_iteration)
    from src.agent import Agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(
            agent.parameters(), lr=args.inner_learning_rate, eps=1e-5
        )
    optimizer = optim.SGD(agent.parameters(), lr=args.inner_learning_rate)
    data_holder = DataHolder(envs_test_set[0], args, device)

    inner_loss, rewards = train_ppo(
        agent,
        envs,
        optimizer,
        data_holder,
        logger=logger,
        num_iteration=args.eval_len,
        is_meta_backbone=False,
        uses_inner_lr=True,
        is_eval=True
    )
    
    return run_name, logger, agent

if __name__ == "__main__":
    run_ids = [f for f in os.listdir(model_directory)]
    runs = [Run(name) for name in run_ids]

    for r in runs:
        if len(sys.argv) >= 2 and r.id != sys.argv[1]:
            continue
        print(r)
        for m in r.model_ids:
            if len(sys.argv) >= 3 and m != int(sys.argv[2]):
                continue
            seed = r.seed
            args, _, device = init_args(seed, -1)
            envs_test_set = init_test_envs_set(args, "Eval")
                    
            for i, _ in enumerate(envs_test_set):
                if len(sys.argv) >= 4 and i != int(sys.argv[3]):
                    continue
                run_name, logger, agent = eval_model_env(r, m, seed, args, device, envs_test_set, i)
                checkpoint(agent, args, m, run_name, logger, should_save_model=False)
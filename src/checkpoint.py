import os
import json

import gymnasium as gym
import torch
from .agent import Agent
from .args import Args
from .logger_base import LoggerBase, MemoryLogger


RESULT_DIR = "experiments/"
MODEL_DIR = "runs/"
STAT_DIR = "stats/"


def save_model(run_name: str, agent: Agent, iteration: int = None):
    model_path = f"{RESULT_DIR}{MODEL_DIR}{run_name}/{iteration}.agent"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")
    return model_path


def load_model(
    run_name: str,
    agent_name: str,
    envs: gym.vector.SyncVectorEnv,
    device: torch.device,
):
    model_path = f"{RESULT_DIR}{MODEL_DIR}{run_name}/{agent_name}.agent"
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    return agent


def save_stats(logger, run_name):
    if type(logger) == MemoryLogger:
        filename = f"{RESULT_DIR}{STAT_DIR}{run_name}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dt = logger.stats
        loc = {}
        for k, v in logger.stats_loc.items():
            loc[k] = [value[1] for value in v]
        dt["loc"] = loc

        with open(filename, "w") as f:
            json.dump(dt, f)
            
        print(f"Statistics saved at {filename}.")


def checkpoint(
    agent: Agent,
    args: Args,
    iteration: int,
    run_name: str,
    logger: LoggerBase,
    should_save_model: bool = True
):
    print(f"Checkpoint at iteration: {iteration}")
    
    if run_name is None:
        return
    
    if should_save_model and args.save_checkpoints:
        save_model(run_name, agent, iteration=iteration)
          
    save_stats(logger, run_name)
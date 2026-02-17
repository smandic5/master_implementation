import sys

import torch.optim as optim
from src.agent import Agent
from src.args import init_args
from src.envs.env_sets import init_train_envs_set
from src.logger import init_logger
from src.ppo.storage import DataHolder
from src.seeds import init_seeds
from src.task_selectors.factory import init_selector
from src.train_maml_ppo import train_maml_ppo


def main(seed, selector_index):

    print(f"Seed: {seed}")
    print(f"Selector: {selector_index}")
        
    args, run_name, device = init_args(seed, selector_index)
    logger = init_logger(args, run_name)
    init_seeds(seed, args.torch_deterministic)

    envs_train_set = init_train_envs_set(args, run_name)
    agent = Agent(envs_train_set[0]).to(device)
    selector = init_selector(
        selector_index,
        envs_train_set,
        logger,
        args,
        agents=[agent for _ in range(len(envs_train_set))],
    )
    optimizer = optim.Adam(
        agent.parameters(), lr=args.meta_learning_rate, eps=1e-5
    )
    data_holder = DataHolder(envs_train_set[0], args, device)

    latest_loss = train_maml_ppo(
        agent,
        selector,
        optimizer,
        data_holder,
        logger=logger,
        run_name=run_name,
    )

    for e in envs_train_set:
        e.close()


if __name__ == "__main__":
    
    TOTAL_SEEDS = 3
    TOTAL_SELECTORS = 6

    if len(sys.argv) >= 2:
        seed = 1
        selector_index = 1
        index = int(sys.argv[1])
        if index >= TOTAL_SEEDS * TOTAL_SELECTORS:
            raise Exception(
                f"Index too high: {index}, where max is {TOTAL_SEEDS * TOTAL_SELECTORS - 1}"
            )
        seed += index // TOTAL_SELECTORS
        selector_index = index % TOTAL_SELECTORS
        if seed >= TOTAL_SEEDS + 1:
            raise Exception(f"Unexpected Seed: {seed}")
        if selector_index >= TOTAL_SELECTORS:
            raise Exception(f"Unexpected Selector: {selector_index}")

        main(seed, selector_index)
        
    else:
        for seed in range(TOTAL_SEEDS):
            for selector_index in range(TOTAL_SELECTORS):
                main(seed, selector_index)

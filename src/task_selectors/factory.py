import gymnasium as gym
import higher
import numpy as np
import scipy.special
import torch
import torch.optim as optim
from ..agent import Agent
from ..args import Args
from ..logger_base import LoggerBase

from .task_selector import (
    HardTaskSelector,
    InsSelector,
    TaskSelector,
    UniformSelector,
    ValueSelector,
    ContextSelector
)


def get_selector_name(index: int) -> str:
    if index == 0:
        return "Uniform Selector"
    elif index == 1:
        return "Hard Task Selector"
    elif index == 2:
        return "INS Selector - Local Dissimilarity"
    elif index == 3:
        return "INS Selector - Generic Dissimilarity"
    elif index == 4:
        return "INS Selector - Local Similarity"
    elif index == 5:
        return "INS Selector - Generic Similarity"
    elif index == 6:
        return "Value Selector - Local Dissimilarity"
    elif index == 7:
        return "Value Selector - Generic Dissimilarity"
    elif index == 8:
        return "Value Selector - Local Similarity"
    elif index == 9:
        return "Value Selector - Generic Similarity"
    elif index == 10:
        return "Context Selector - Local Dissimilarity"
    elif index == 11:
        return "Context Selector - Generic Dissimilarity"
    elif index == 12:
        return "Context Selector - Local Similarity"
    elif index == 13:
        return "Context Selector - Generic Similarity"
    else:
        raise Exception(f"Unrecognized selector index: {index}")


def init_selector(
    index: int,
    envs: list[gym.vector.SyncVectorEnv],
    logger: LoggerBase,
    args: Args,
    agents: list[Agent] = None,
) -> TaskSelector:
    disimilarity = None
    from_last = None
    if index == 0:
        cls = UniformSelector
    elif index == 1:
        cls = HardTaskSelector
    elif index == 2:
        cls = InsSelector
        disimilarity = True
        from_last = True
    elif index == 3:
        cls = InsSelector
        disimilarity = True
        from_last = False
    elif index == 4:
        cls = InsSelector
        disimilarity = False
        from_last = True
    elif index == 5:
        cls = InsSelector
        disimilarity = False
        from_last = False
    elif index == 6:
        cls = ValueSelector
        disimilarity = True
        from_last = True
    elif index == 7:
        cls = ValueSelector
        disimilarity = True
        from_last = False
    elif index == 8:
        cls = ValueSelector
        disimilarity = False
        from_last = True
    elif index == 9:
        cls = ValueSelector
        disimilarity = False
        from_last = False
    elif index == 10:
        cls = ContextSelector
        disimilarity = True
        from_last = True
    elif index == 11:
        cls = ContextSelector
        disimilarity = True
        from_last = False
    elif index == 12:
        cls = ContextSelector
        disimilarity = False
        from_last = True
    elif index == 13:
        cls = ContextSelector
        disimilarity = False
        from_last = False
    else:
        raise Exception(f"Unrecognized selector index: {index}")
    return cls(
        envs,
        logger=logger,
        disimilarity=disimilarity,
        from_last=from_last,
        agents=agents,
        args=args,
    )

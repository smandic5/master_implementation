import gymnasium as gym
import numpy as np
from ..args import Args
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv


class HalfCheetahMetaWrapper(gym.Wrapper):
    def __init__(self, env, target_velocity: float):
        super().__init__(env)
        self.target_velocity = target_velocity
        self.velocity_reward_weight = 0.1

    def step(self, action):
        envs: HalfCheetahEnv = self.env

        observation, reward, terminated, truncated, info = envs.step(action)
        # return observation, reward, terminated, truncated, info

        reward_ctrl = info["reward_ctrl"]
        x_velocity = info["x_velocity"]
        #forward_reward = -1.0 * abs(x_velocity - self.target_velocity)
        #forward_reward *= self.velocity_reward_weight
        
        forward_reward = -2 * max(0, x_velocity - self.target_velocity)
        
        info["velocity_reward"] = forward_reward
        final_reward = forward_reward + x_velocity + 0.5 * reward_ctrl
        
        return observation, final_reward, terminated, truncated, info

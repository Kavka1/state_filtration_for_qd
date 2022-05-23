from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.noise_obs_wrapper import Noise_Obs_Wrapper
from state_filtration_for_qd.env.common import call_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        self.model.load_model(model_path)
        self.num_episode = num_episode

    def set_env(self, env: gym.Env)-> None:
        self.env = env

    def rollout(self) -> float:
        total_reward = 0
        for _ in range(self.num_episode):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.model.act(
                    torch.from_numpy(obs).float(),
                    False
                )
                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode



def main(path: str, remark: str, noise_index: List[int]) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['num_primitive']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/primitive_{k}_{remark}'
        all_workers.append(Worker(model_path, config['model_config'], 20))

    for noise_scale in list(np.linspace(0, 1, 11)):
        all_env = [Noise_Obs_Wrapper(
            call_env(config['env_config']['env_name']), 'Gaussian', noise_scale, noise_index
        ) for _ in range(num_primitive)]
        ray.put(all_env)
        remotes = [worker.set_env(env) for worker, env in zip(all_workers, all_env)]
        ray.get(remotes)

        rollout_remote = [worker.rollout() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)

        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = {'primitive {j}': all_primitive_scores[j] for j in range(num_primitive)}
        print(f"noise scale {noise_scale}:\n   {all_primitive_scores}\n    best primitive: {best_primitive_index}")
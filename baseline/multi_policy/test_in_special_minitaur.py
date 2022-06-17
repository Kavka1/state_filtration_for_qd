from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_special_minitaur_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, is_actual_model:bool, noise_std: float, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            model_config['policy_activation'],
        )
        self.model.load_model(model_path)
        self.env = call_special_minitaur_env(is_actual_model, noise_std)
        self.num_episode = num_episode
    
    def set_env(self, is_actual_model:bool, noise_scale: float)-> None:
        self.env = call_special_minitaur_env(is_actual_model, noise_scale)

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


def test_in_actual_model(path: str, remark: str, csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['model_config']['z_dim']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], True, 0, 20))

    score_dict = {f'primitive {k}': [] for k in range(num_primitive)}

    rollout_remote = [worker.rollout.remote() for worker in all_workers]
    all_primitive_scores = ray.get(rollout_remote)
    best_primitive_index = np.argmax(all_primitive_scores)
    all_primitive_scores = [all_primitive_scores[j] for j in range(num_primitive)]
        
    for k in range(num_primitive):
        print(f"    Primitive {k}: {all_primitive_scores[k]}")
    print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

    for k in range(num_primitive):
        score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)


def test_in_noisy_obs(path: str, remark: str, csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['model_config']['z_dim']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], False, 0, 20))

    noise_range = [round(j * 0.1 + 0, 2) for j in range(16)]
    score_dict = {'noise scale': noise_range}
    score_dict.update({f'primitive {k}': [] for k in range(num_primitive)})


    for noise_scale in noise_range:
        remotes = [worker.set_env.remote(False, noise_scale) for worker in all_workers]
        ray.get(remotes)

        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)

        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = [all_primitive_scores[j] for j in range(num_primitive)]
        
        print(f'\nnoise scale {noise_scale}')
        for k in range(num_primitive):
            print(f"    Primitive {k}: {all_primitive_scores[k]}")
        print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

        for k in range(num_primitive):
            score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    for seed in [10, 20, 30, 40, 50]:
        test_in_actual_model(   
            path= f'/home/xukang/Project/state_filtration_for_qd/resutls_for_multi_policy/Minitaur-{seed}/',
            remark='best',
            csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/Minitaur_actual-{seed}.csv'
        )
        test_in_noisy_obs(   
            path= f'/home/xukang/Project/state_filtration_for_qd/resutls_for_multi_policy/Minitaur-{seed}/',
            remark='best',
            csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/Minitaur_noise-{seed}.csv'
        )
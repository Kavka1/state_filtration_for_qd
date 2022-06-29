from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, damaged_leg: int, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            model_config['policy_activation'],
        )
        self.model.load_model(model_path)
        self.env = call_env(
            {
                'env_name': 'Minitaur',
                'missing_obs_info': {
                    'missing_angle':    [],
                }
            }
        )
        self.num_episode = num_episode

        if damaged_leg == 0:
            self.damage_a_idx = [0,4]
        elif damaged_leg == 1:
            self.damage_a_idx = [1,5]
        elif damaged_leg == 2:
            self.damage_a_idx = [2,6]
        else:
            self.damage_a_idx = [3,7]
    
    def rollout(self) -> float:
        total_reward = 0
        for _ in range(self.num_episode):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.model.act(
                    torch.from_numpy(obs).float(),
                    False
                ).detach().numpy()
                
                # broke a motor
                a[self.damage_a_idx[0]] = 0.
                a[self.damage_a_idx[1]] = 0.

                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode


def test_in_damaged_leg(path: str, remark: str, csv_path: str, damage_leg: int) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['num_primitive']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], damage_leg, 10))

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


if __name__ == "__main__":
    for damaged_leg in [1, 2, 3, 4]:
        for seed in [
            #10, 20, 30, 40, 50
            60, 70, 80
        ]:
            test_in_damaged_leg(   
                path= f'/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Minitaur-missing_angle_1_2_3_4-{seed}/',
                remark='best',
                csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Minitaur_damage_leg_{damaged_leg}-{seed}.csv',
                damage_leg=damaged_leg,
            )
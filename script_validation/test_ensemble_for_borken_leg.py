from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_env, call_broken_leg_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, env_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        self.model.load_model(model_path)
        self.env = call_broken_leg_env(env_config)
        self.num_episode = num_episode

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


def main(path: str, remark: str, env_config: Dict, csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['num_primitive']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], env_config, 20))

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
    for env in ['Walker']:
        if env == 'Hopper':
            path_mark = 'missing_leg_1'
        elif env == 'Walker':
            path_mark = 'missing_leg_1'
            
        for seed in [40, 50]:
            for leg_foot in [[0,1], [1,0]]:
                if leg_foot == [0,1]:
                    csv_mark = 'right_leg'
                else:
                    csv_mark = 'right_foot'

                main(   
                    path= f'/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/{env}-{path_mark}-{seed}/',
                    remark='best',
                    env_config={
                        'env_name': env,
                        'broken_leg_info': {
                            'is_left_leg': False,
                            'leg_jnt_scale': leg_foot[0],
                            'foot_jnt_scale': leg_foot[1],
                        }
                    },
                    csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{csv_mark}-{seed}.csv'
                )
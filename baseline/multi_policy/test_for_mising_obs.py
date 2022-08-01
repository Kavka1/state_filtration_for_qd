from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.missing_obs_wrapper import Missing_Obs_Wrapper
from state_filtration_for_qd.env.common import call_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            'Tanh'
        )
        self.model.load_model(model_path)
        self.num_episode = num_episode

    def set_env(self, env_config: Dict, obs_index: List[int])-> None:
        self.env = Missing_Obs_Wrapper(call_env(env_config), obs_index)

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



def main(path: str, remark: str, obs_index: List[int], csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['model_config']['z_dim']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], 20))

    score_dict = {}
    score_dict.update({f'primitive {k}': [] for k in range(num_primitive)})

    remotes = [worker.set_env.remote(config['env_config'], obs_index) for worker in all_workers]
    ray.get(remotes)

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


if __name__ == '__main__':
    env = 'Walker'
    
    for obs_index in [
        #[2,3,4,8,9,10],
        #[2,3,4,11,12,13],
        #list(range(1,4)),
        #list(range(4,7)),
        #list(range(7,10)),
        #list(range(10,13)),
        [2,3,4],
        [5,6,7]
    ]:
        if obs_index == [2,3,4]:
            csv_mark = 'coord_2'
        elif obs_index == [5,6,7]:
            csv_mark = 'coord_3'

        for seed in [10, 20, 30, 40, 50, 60, 70, 80]:
            main(
                path=f'/home/xukang/Project/state_filtration_for_qd/results_for_multi/{env}-{seed}/',
                remark='best',
                obs_index=obs_index,
                csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}-defective_sensor-{csv_mark}-{seed}.csv'
            )
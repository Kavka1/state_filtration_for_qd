from typing import List, Dict, Tuple
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
    def __init__(self, model_path: str, model_config: Dict, env_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            'Tanh',
        )
        self.model.load_model(model_path)
        self.env = call_env(env_config)
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
                ).detach().numpy()
                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode



def main(path_root: str, all_seeds: List[str], remark: str, env_config: Dict, csv_path: str) -> None:
    test_path = path_root + f'-{all_seeds[0]}/'
    with open(test_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    num_primitive = config['num_primitive']
    
    # data structure for return across all seeds and primitives
    return_across_seeds_and_primitive = {f'seed {s}': [] for s in all_seeds}
    for seed in all_seeds:
        path = path_root + f'-{seed}/'
        with open(test_path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        all_workers = []
        for k in range(num_primitive):
            model_path = path + f'model/policy_{k}_{remark}'
            all_workers.append(Worker.remote(model_path, config['model_config'], env_config, 20))

        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)

        for primitive_idx in range(num_primitive):
            print(f'seed {seed} primitive {primitive_idx}: {all_primitive_scores[primitive_idx]}')
            return_across_seeds_and_primitive[f'seed {seed}'].append(float(all_primitive_scores[primitive_idx]))

    score_df = pd.DataFrame(return_across_seeds_and_primitive)
    #score_df.to_csv(csv_path, index=False)



if __name__ == '__main__':
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]
    all_seeds = ['50']

    all_path_roots = {
        'Walker':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1',
        'Hopper':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Hopper-missing_leg_1',
        'Ant':      '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_leg_1_2_3_4',
        'Minitaur': '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/new_trdeoff-Minitaur-missing_angle_1_2_3_4'
    }
    all_csv_paths = {
        'Walker':   '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Walker_return_dist.csv',
        'Hopper':   '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Hopper_return_dist.csv',
        'Ant':      '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Ant_return_dist.csv',
        'Minitaur': '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/new_trdeoff-Minitaur_return_dist.csv'
    }

    for env in [
        #'Walker', 'Hopper', 'Ant', 
        'Minitaur'
    ]:
        if env in ['Walker', 'Hopper', 'Ant']:
            env_config = {
                'env_name': env,
                'missing_obs_info': {
                    'missing_coord':    [],
                    'missing_joint':    [],
                    'missing_leg':      []
                }
            }
        else:
            env_config = {
                'env_name': 'Minitaur',
                'missing_obs_info': {
                    'missing_angle':    [],
                }
            }

        main(
            all_path_roots[env],
            all_seeds,
            'best',
            env_config,
            all_csv_paths[env]
        )
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
        all_rewards = []
        for _ in range(self.num_episode):
            epi_r = 0
            done = False
            obs = self.env.reset()
            while not done:
                a = self.model.act(
                    torch.from_numpy(obs).float(),
                    False
                ).detach().numpy()
                obs, r, done, info = self.env.step(a)
                epi_r += r
            all_rewards.append(epi_r)
        return all_rewards



def main(path_root: str, all_seeds: List[str], remark: str, env_config: Dict, csv_path: str, num_episode: int, cvar_frac: float) -> None:
    test_path = path_root + f'-{all_seeds[0]}/'
    with open(test_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    num_primitive = config['model_config']['z_dim']
    
    # data structure for return across all seeds and primitives
    return_across_seeds_and_primitive = {f'seed {s}': [] for s in all_seeds}
    for seed in all_seeds:
        path = path_root + f'-{seed}/'
        with open(test_path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        all_workers = []
        for k in range(num_primitive):
            model_path = path + f'model/policy_{k}_{remark}'
            all_workers.append(Worker.remote(model_path, config['model_config'], env_config, num_episode))

        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_score_seq = ray.get(rollout_remote)

        for primitive_idx in range(num_primitive):
            score_seq = all_primitive_score_seq[primitive_idx]
            score_seq.sort()
            cvar = np.mean(score_seq[:int(cvar_frac * num_episode)])
            print(f'seed {seed} primitive {primitive_idx}: {cvar}')
            return_across_seeds_and_primitive[f'seed {seed}'].append(float(cvar))

    score_df = pd.DataFrame(return_across_seeds_and_primitive)
    score_df.to_csv(csv_path, index=False)



if __name__ == '__main__':
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]

    all_path_roots = {
        'Walker':   '/home/xukang/Project/state_filtration_for_qd/results_for_multi/Walker',
        'Hopper':   '/home/xukang/Project/state_filtration_for_qd/results_for_multi/Hopper',
        'Ant':      '/home/xukang/Project/state_filtration_for_qd/results_for_multi/Ant',
        'Minitaur': '/home/xukang/Project/state_filtration_for_qd/results_for_multi/Minitaur'
    }
    all_csv_paths = {
        'Walker':   '/home/xukang/Project/state_filtration_for_qd/statistic/multi/Walker_cvar25_dist.csv',
        'Hopper':   '/home/xukang/Project/state_filtration_for_qd/statistic/multi/Hopper_cvar25_dist.csv',
        'Ant':      '/home/xukang/Project/state_filtration_for_qd/statistic/multi/Ant_cvar25_dist.csv',
        'Minitaur': '/home/xukang/Project/state_filtration_for_qd/statistic/multi/Minitaur_cvar25_dist.csv'
    }

    for env in [
        #'Walker', 'Hopper', 
        'Ant', 
        #'Minitaur'
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
            all_csv_paths[env],
            200,
            0.25
        )
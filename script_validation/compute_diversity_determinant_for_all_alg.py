from typing import List, Dict, Tuple
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_env



def state_collection(env_config: Dict, num_collect_state: int) -> np.array:
    env = call_env(env_config)
    state_batch = []
    
    obs = env.reset()
    while len(state_batch) < num_collect_state:
        a = env.action_space.sample()
        next_obs, r, done, _ = env.step(a)
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        state_batch.append(obs)
    state_batch = np.stack(state_batch, 0)
    return state_batch


def compuate_diversity_determinant(pop_action_batch: List[torch.tensor]) -> float:
    action_embedding = [action.flatten() for action in pop_action_batch]
    embedding = torch.stack(action_embedding, 0)
    left = embedding.unsqueeze(0).expand(embedding.size(0),-1,-1)
    right = embedding.unsqueeze(1).expand(-1, embedding.size(0),-1)
    matrix = torch.exp(- torch.square(left - right)).sum(-1) / 2
    determinant = torch.logdet(matrix)
    return determinant.detach().item()



def main(path_root: str, all_seeds: List[str], remark: str, env_config: Dict, csv_path: str, a_dim: List[int] = None) -> None:
    state_batch = state_collection(env_config, 2000)
    state_batch_tensor = torch.from_numpy(state_batch).float().to(torch.device('cuda'))

    all_diversity_score = {f'seed {seed}': [] for seed in all_seeds}

    for seed in all_seeds:
        path = path_root + f'-{seed}/'
        with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_config = config['model_config']
        
        pop_action_batch = []
        for k in range(10):
            model_path = path + f'model/policy_{k}_{remark}'
            model = FixStdGaussianPolicy(
                model_config['o_dim'],
                model_config['a_dim'],
                model_config['policy_hidden_layers'],
                model_config['action_std'],
                'Tanh',
            ).to(torch.device('cuda'))
            model.load_model(model_path)
            action_batch = model(state_batch_tensor).mean
            if a_dim:
                action_batch = action_batch.index_select(-1, torch.tensor(a_dim).to('cuda'))
            pop_action_batch.append(action_batch)

        diversity_score = compuate_diversity_determinant(pop_action_batch)
        all_diversity_score[f'seed {seed}'].append(diversity_score)

    score_df = pd.DataFrame(all_diversity_score)
    if a_dim:
        score_df.to_csv(csv_path + f'a_dim_{a_dim}.csv', index=False)
    else:
        score_df.to_csv(csv_path + '.csv', index=False)


if __name__ == '__main__':
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]
    
    a_dim = [0,1,2]

    for alg in [
        'ensemble',
        'dvd',
        'smerl_ppo',
        'multi'
    ]:
        if alg == 'ensemble':
            all_path_roots = {
                'Walker':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1',
                'Hopper':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Hopper-missing_leg_1',
                'Ant':      '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_leg_1_2_3_4',
                'Minitaur': '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/new_trdeoff-Minitaur-missing_angle_1_2_3_4'
            }
            all_csv_paths = {
                'Walker':   '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Walker_diversity_score',
                'Hopper':   '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Hopper_diversity_score',
                'Ant':      '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Ant_diversity_score',
                'Minitaur': '/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/Minitaur_diversity_score'
            }
        else:
            all_path_roots = {
                'Walker':   f'/home/xukang/Project/state_filtration_for_qd/results_for_{alg}/Walker',
                'Hopper':   f'/home/xukang/Project/state_filtration_for_qd/results_for_{alg}/Hopper',
                'Ant':      f'/home/xukang/Project/state_filtration_for_qd/results_for_{alg}/Ant',
                'Minitaur': f'/home/xukang/Project/state_filtration_for_qd/results_for_{alg}/Minitaur'
            }
            all_csv_paths = {
                'Walker':   f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/Walker_diversity_score',
                'Hopper':   f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/Hopper_diversity_score',
                'Ant':      f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/Ant_diversity_score',
                'Minitaur': f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/Minitaur_diversity_score'
            }

        for env in [
            'Walker', 
            #'Hopper', 
            #'Ant', 
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
                a_dim
            )
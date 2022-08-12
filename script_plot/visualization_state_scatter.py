import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_env



def state_collection(env_config: Dict, policy: FixStdGaussianPolicy, num_collect_state: int, bernoulli: float, select_s_dim: List[int]) -> List:
    env = call_env(env_config)
    all_state = []
    
    obs = env.reset()
    while len(all_state) < num_collect_state:
        a = policy.act(torch.from_numpy(obs).to('cuda').float(), False).detach().cpu().numpy()
        next_obs, r, done, _ = env.step(a)
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        if random.uniform(0, 1) < bernoulli:
            all_state.append(obs[select_s_dim].tolist())
    return all_state



def main(all_path_root: str, all_seeds: List[str], remark: str, env_config: Dict, num_state: int, bernoulli: float, selected_s_dim: List[int], all_title: List[str]) -> None:
    #fig, axs = plt.subplots(1, len(all_path_root), tight_layout=True, figsize=(15, 4))
    
    std_1_of_algs_across_seed = []
    std_2_of_algs_across_seed = []

    for seed in all_seeds:
        std_1_across_alg = []
        std_2_across_alg = []

        for i, path_root in enumerate(all_path_root):
            path = path_root + f'-{seed}/'
            with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_config = config['model_config']
            
            df_for_pop = []

            mean_1_across_policy = []
            mean_2_across_policy = []

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
                s_batch_k = state_collection(env_config, model, num_state, bernoulli, selected_s_dim)    
                s_1_batch = [s[0] for s in s_batch_k]
                s_2_batch = [s[1] for s in s_batch_k]

                df_for_pop.append(pd.DataFrame({
                    'policy': [k for _ in range(num_state)],
                    f'state_{selected_s_dim[0]}': s_1_batch,
                    f'state_{selected_s_dim[1]}': s_2_batch
                }))
                print(f'Finished alg {i} policy {k}')

                mean_1_across_policy.append(np.mean(s_1_batch))
                mean_2_across_policy.append(np.mean(s_2_batch))

            std_1_across_alg.append(np.std(mean_1_across_policy))
            std_2_across_alg.append(np.std(mean_2_across_policy))

            df_for_pop = pd.concat(df_for_pop)
            #sns.kdeplot(
            #    data= df_for_pop,
            #    x= f'state_{selected_s_dim[0]}',
            #    y= f'state_{selected_s_dim[1]}',
            #    hue= 'policy',
            #    ax= axs[i]
            #)
            #axs[i].set_title(f"{env_config['env_name']}-{all_title[i]}-{seed}")
            #axs[i].legend().remove()

        std_1_of_algs_across_seed.append(std_1_across_alg)
        std_2_of_algs_across_seed.append(std_2_across_alg)

        print(f"seed: {seed} std of population in state 1: {std_1_across_alg}\nstd of population in state 2: {std_2_across_alg}")

    print(f"std_1 across seeds: {np.array(std_1_of_algs_across_seed).mean(0)}-{np.array(std_1_of_algs_across_seed).std(0)} \n\
            std_2 across seeds: {np.array(std_2_of_algs_across_seed).mean(0)}-{np.array(std_2_of_algs_across_seed).std(0)}")
        #plt.show()



if __name__ == '__main__':
    seed = 10
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]
    env = 'Walker'
    selected_s_dim = [5,6]
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

    all_path_root = []
    for alg in [
        'ensemble',
        'dvd',
        'smerl_ppo',
        'multi'
    ]:
        if alg == 'ensemble':
            all_path_root.append(
                {
                    'Walker':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1',
                    'Hopper':   '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Hopper-missing_leg_1',
                    'Ant':      '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_leg_1_2_3_4',
                    'Minitaur': '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/new_trdeoff-Minitaur-missing_angle_1_2_3_4'
                }[env]
            )
        else:
            all_path_root.append(f'/home/xukang/Project/state_filtration_for_qd/results_for_{alg}/{env}')

    all_title = [
        'DiR',
        'DvD',
        'SMERL',
        'Multi'
    ]
    
    main(
        all_path_root,
        all_seeds,
        #seed,
        'best',
        env_config,
        1000,
        1,
        selected_s_dim,
        all_title
    )

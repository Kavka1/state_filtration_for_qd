from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
import gym

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]



def check_if_all_zero(array: List[float]) -> bool:
    for item in array:
        if item != 0:
            return False
    return True


def main(all_path: str, remark: str, num_episode: int) -> None:
    sns.set_style('white')
    sns.set_palette(sns.color_palette(flatui))
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(16, 3))
    #sns.despine(fig, ax=ax)

    for j, path in enumerate(all_path):

        heat_array = np.zeros((10, 10))

        with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_config = config['model_config']
        env = gym.make('Walker2d-v4')
        
        all_Pis  = []
        all_primitive_idx = [i for i in range(10)]

        for k in all_primitive_idx:
            policy = FixStdGaussianPolicy(
                model_config['o_dim'],
                model_config['a_dim'],
                model_config['policy_hidden_layers'],
                model_config['action_std'],
                'Tanh'
            )
            policy.load_model(path + f'model/policy_{k}_{remark}')
            all_Pis.append(policy)
    
        df = []
        for i in range(len(all_Pis)):
            policy = all_Pis[i]
            left_foot_contact_frac  = []
            right_foot_contact_frac = []

            for episode in range(num_episode):
                left_foot_contact_count   = 0
                right_foot_contact_count  = 0

                obs = env.reset()
                done = False
                step = 0
                episode_r = 0
                while not done:
                    obs = torch.from_numpy(obs).float()
                    cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
                    a = policy.act(obs, False).detach().numpy()
                    obs_, r, done, info = env.step(a)
                    obs = obs_
                    episode_r += r
                    step += 1

                    # for left leg
                    if not check_if_all_zero(cfrc_ext[7]):
                        left_foot_contact_count += 1
                    # for right leg
                    if not check_if_all_zero(cfrc_ext[4]):
                        right_foot_contact_count += 1

                l_contact_frac = left_foot_contact_count / step
                r_contact_frac = right_foot_contact_count / step

                left_foot_contact_frac.append(l_contact_frac)
                right_foot_contact_frac.append(r_contact_frac)

                heat_array[int(l_contact_frac * 10), int(r_contact_frac * 10)] = episode_r

            
        ax = axs[j]
        sns.heatmap(
            data= heat_array,
            vmin=0,
            #vmax=5000,
            cmap="YlGnBu",
            ax=ax
        )

        #for _, s in ax.spines.items():
        #    s.set_linewidth(1.1)

        #ax.set_ylim([0,0.5])
        #ax.set_xlim([0,0.5])
        #ax.legend().remove()
        #ax.set_xlabel('Time proportion of left foot contacting the ground', fontsize=12)
        #ax.set_ylabel('Time proportion of right foot contacting the ground', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        ax.set_title('')
        #ax.set_title()
    
    plt.show()


if __name__ == '__main__':
    all_exp_path = [
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-50/',
        '/home/xukang/Project/state_filtration_for_qd/results_for_smerl_ppo/Walker-50/',
        '/home/xukang/Project/state_filtration_for_qd/results_for_dvd/Walker-50/',
        '/home/xukang/Project/state_filtration_for_qd/results_for_multi/Walker-50/'
    ]

    chosen_mark         = 'best'
    main(
        all_exp_path,
        remark=chosen_mark,
        num_episode= 10
    )
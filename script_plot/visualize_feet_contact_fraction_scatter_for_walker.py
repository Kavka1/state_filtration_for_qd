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


def main(path: str, remark: str, num_episode: int) -> None:
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

    sns.set_style('white')
    sns.set_palette(sns.color_palette(flatui))

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(5.5, 5))
    sns.despine(fig, ax=ax)
    
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
            while not done:
                obs = torch.from_numpy(obs).float()
                cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
                a = policy.act(obs, False).detach().numpy()
                obs_, r, done, info = env.step(a)
                obs = obs_

                step += 1

                # for left leg
                if not check_if_all_zero(cfrc_ext[7]):
                    left_foot_contact_count += 1
                # for right leg
                if not check_if_all_zero(cfrc_ext[4]):
                    right_foot_contact_count += 1

            left_foot_contact_frac.append(left_foot_contact_count / step)
            right_foot_contact_frac.append(right_foot_contact_count / step)

        df.append(
            pd.DataFrame({
                'left foot contact proportion'  : left_foot_contact_frac,
                'right foot contact proportion' : right_foot_contact_frac,
                'policy'                        : [f'policy {i}'] * num_episode
            })
        )
    
    df = pd.concat(df)
    sns.scatterplot(
        data    = df,
        x       = 'left foot contact proportion',
        y       = 'right foot contact proportion',
        hue     = 'policy',
        palette = 'Dark2_r',
        s       = 30,
        ax      = ax,
    )

    for _, s in ax.spines.items():
        s.set_linewidth(1.1)

    ax.set_ylim([0,0.5])
    ax.set_xlim([0,0.5])
    ax.legend().remove()
    ax.set_xlabel('Time proportion of left foot contacting the ground', fontsize=12)
    ax.set_ylabel('Time proportion of right foot contacting the ground', fontsize=12)
    ax.set_title('')
    #ax.set_title()
    plt.show()


if __name__ == '__main__':
    chosen_exp_path     = '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-50/'
    chosen_mark         = 'best'
    main(
        path= chosen_exp_path,
        remark=chosen_mark,
        num_episode= 20
    )
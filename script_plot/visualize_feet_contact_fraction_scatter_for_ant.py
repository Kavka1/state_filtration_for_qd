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
    env = gym.make('Ant-v4')
    
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
        foot_1_contact_frac  = []
        foot_2_contact_frac = []

        for episode in range(num_episode):
            foot_1_contact_count   = 0
            foot_2_contact_count  = 0

            obs = env.reset()
            done = False
            step = 0
            reward = 0
            while not done:
                #env.render()
                obs = np.concatenate([obs, np.zeros(84)])

                obs = torch.from_numpy(obs).float()
                cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
                a = policy.act(obs, False).detach().numpy()
                obs_, r, done, info = env.step(a)
                obs = obs_

                step += 1
                reward += r

                # for left leg
                if not check_if_all_zero(cfrc_ext[4]):
                    foot_1_contact_count += 1
                # for right leg
                if not check_if_all_zero(cfrc_ext[7]):
                    foot_2_contact_count += 1

            print(f'step {step} reward {reward}')

            foot_1_contact_frac.append(foot_1_contact_count / step)
            foot_2_contact_frac.append(foot_2_contact_count / step)

        df.append(
            pd.DataFrame({
                'foot 1 contact proportion'  : foot_1_contact_frac,
                'foot 2 contact proportion' : foot_2_contact_frac,
                'policy'                        : [f'policy {i}'] * num_episode
            })
        )
    
    df = pd.concat(df)
    sns.scatterplot(
        data    = df,
        x       = 'foot 1 contact proportion',
        y       = 'foot 2 contact proportion',
        hue     = 'policy',
        palette = 'Dark2_r',
        s       = 30,
        ax      = ax,
    )

    for _, s in ax.spines.items():
        s.set_linewidth(1.1)

    ax.set_ylim([0,0.3])
    ax.set_xlim([0,0.3])
    ax.legend().remove()
    ax.set_xlabel('Time proportion of foot 1 contacting the ground', fontsize=12)
    ax.set_ylabel('Time proportion of foot 2 contacting the ground', fontsize=12)
    ax.set_title('')
    #ax.set_title()
    plt.show()


if __name__ == '__main__':
    chosen_exp_path     = '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_leg_1_2_3_4-70/'
    chosen_mark         = 'best'
    main(
        path= chosen_exp_path,
        remark=chosen_mark,
        num_episode= 20
    )
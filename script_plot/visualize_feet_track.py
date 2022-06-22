from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import gym

from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy


def check_if_all_zero(array: List[float]) -> bool:
    for item in array:
        if item != 0:
            return False
    return True


def main(path: str, remark: str, chosen_ptimitive: List[int], episode_length=200) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env = gym.make('Walker2d-v4')
    
    all_Pis  = []
    for k in range(chosen_primitive):
        policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        policy.load_model(path + f'model/policy_{k}_{remark}')
        all_Pis.append(policy)

    fig, axs = plt.subplots(nrows=len(chosen_primitive), ncols=1, sharex=True, tight_layout=True)
    for i, ax in enumerate(axs):
        policy = all_Pis[i]
        left_foot_contact   = []
        right_foot_contact  = []

        obs = env.reset()
        for step in range(episode_length):
            obs = torch.from_numpy(obs).float()
            cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
            a = policy.act(obs, False).detach().numpy()
            obs_, r, done, info = env.step(a)
            next_cfrc_ext = copy(env.unwrapped_data.cfrc_ext)
            
            # for left leg
            if not check_if_all_zero(cfrc_ext[7]) and check_if_all_zero(next_cfrc_ext[7]):
                left_foot_contact.append((step, 0))
            elif len(left_foot_contact)!=0 and check_if_all_zero(cfrc_ext[7]) and check_if_all_zero(next_cfrc_ext[7]):
                left_foot_contact[-1][1] += 1
            
            # for right leg
            if not check_if_all_zero(cfrc_ext[4]) and check_if_all_zero(next_cfrc_ext[4]):
                left_foot_contact.append((step, 0))
            elif len(right_foot_contact)!=0 and check_if_all_zero(cfrc_ext[4]) and check_if_all_zero(next_cfrc_ext[4]):
                left_foot_contact[-1][1] += 1

        ax.broken_barh(left_foot_contact, (5, 10), facecolors='tab:blue')
        ax.broken_barh(right_foot_contact, (20, 10), facecolors='tab:orange')

        ax.set_ylim(5, 30)
        ax.set_xlim(0, episode_length)
        ax.set_xlabel('time steps')
        ax.set_yticks([10, 25], labels=['left foot', 'right foot'])

    plt.show()


if __name__ == '__main__':
    chosen_exp_path     = '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-10/'
    chosen_mark         = 'best'
    chosen_primitive    = [0, 5, 6, 7]
    
    main(
        path= chosen_exp_path,
        remark=chosen_mark,
        chosen_ptimitive=chosen_primitive,
        episode_length=200
    )
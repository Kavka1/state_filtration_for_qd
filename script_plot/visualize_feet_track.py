from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from copy import copy
import gym

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy


all_color_pairs = [
    ['orange', 'darkorange'],
    ['skyblue', 'deepskyblue'],
    ['limegreen', 'seagreen'],
    ['pink', 'deeppink']
]


def check_if_all_zero(array: List[float]) -> bool:
    for item in array:
        if item != 0:
            return False
    return True


def main(path: str, remark: str, chosen_primitive: List[int], episode_length=200) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env = gym.make('Walker2d-v4')
    
    all_Pis  = []
    for k in chosen_primitive:
        policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            'Tanh'
        )
        policy.load_model(path + f'model/policy_{k}_{remark}')
        all_Pis.append(policy)

    fig, axs = plt.subplots(nrows=len(chosen_primitive), ncols=1, sharex=True, tight_layout=True, figsize=(7, 3))
    for i, ax in enumerate(axs):
        policy = all_Pis[i]
        left_foot_contact   = []
        right_foot_contact  = []

        obs = env.reset()
        for step in range(episode_length):
            env.render()
            obs = torch.from_numpy(obs).float()
            cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
            a = policy.act(obs, False).detach().numpy()
            obs_, r, done, info = env.step(a)
            next_cfrc_ext = copy(env.unwrapped.data.cfrc_ext)
            obs = obs_

            # for left leg
            if check_if_all_zero(cfrc_ext[7]) and not check_if_all_zero(next_cfrc_ext[7]):
                left_foot_contact.append([step, 0])
            elif len(left_foot_contact)!=0 and not check_if_all_zero(cfrc_ext[7]) and not check_if_all_zero(next_cfrc_ext[7]):
                left_foot_contact[-1][1] += 1
            
            # for right leg
            if check_if_all_zero(cfrc_ext[4]) and not check_if_all_zero(next_cfrc_ext[4]):
                right_foot_contact.append([step, 1])
            elif len(right_foot_contact)!=0 and not check_if_all_zero(cfrc_ext[4]) and not check_if_all_zero(next_cfrc_ext[4]):
                right_foot_contact[-1][1] += 1


        for j in reversed(range(len(left_foot_contact))):
            if left_foot_contact[j][-1] == 1:
                item = left_foot_contact[j]
                left_foot_contact.remove(item)
        for j in reversed(range(len(right_foot_contact))):
            if right_foot_contact[j][-1] == 1:
                item = right_foot_contact[j]
                right_foot_contact.remove(item)

        ax.broken_barh(left_foot_contact, (18, 10), facecolors=all_color_pairs[i][0])
        ax.broken_barh(right_foot_contact, (5, 10), facecolors=all_color_pairs[i][1])

        ax.set_ylim(5, 28)
        ax.set_xlim(0, episode_length)
        ax.set_yticks([23, 10], labels=['LF', 'RF'], fontsize=12)

    axs[-1].set_xlabel('time step', fontsize=13)
    plt.show()


if __name__ == '__main__':
    chosen_exp_path     = '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-10/'
    chosen_mark         = 'best'
    chosen_primitive    = [0, 6, 7, 8]
    
    main(
        path= chosen_exp_path,
        remark=chosen_mark,
        chosen_primitive=chosen_primitive,
        episode_length=600
    )
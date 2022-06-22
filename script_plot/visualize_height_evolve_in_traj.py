from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy


def main(path: str, remark: str, num_traj_per_prim: int, chosen_primitive = [0, 6, 7, 8]) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env_config   = config['env_config']
    
    env = call_env(env_config)
    env.apply_missing_obs = False            # output the clipped observation
    
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


    height_traj_dict = dict()
    reward_dict = dict()
    for i, policy in enumerate(all_Pis):
        traj_seq = []
        reward_seq = []
        for j in range(num_traj_per_prim):
            height_traj = []
            obs = env.reset()
            done = False
            reward = 0
            while not done:
                action = policy.act(torch.from_numpy(obs).float(), False).detach().numpy()
                obs_, r, done, info = env.step(action)
                height_traj.append(obs[0])      # must guarantee obs[0] is the value at global y-coordinate
                obs = obs_
                reward += r
            #if len(height_traj) > min_traj_len:
            traj_seq.append(height_traj)
            reward_seq.append(reward)
        height_traj_dict.update({f'primitive {i}': traj_seq})
        reward_dict.update({f'primitive {i}': reward_seq})

    all_df = []
    for key in list(height_traj_dict.keys()):
        df_per_prim = []
        for j in range(len(height_traj_dict[key])):
            traj = height_traj_dict[key][j]
            reward = reward_dict[key][j]

            if reward > 3000:
                ret = 'ret > 3000'
            elif 2000 < reward and reward <= 3000:
                ret = '2000 < ret < 3000'
            elif 1000 < reward and reward <= 2000:
                ret = '1000 < ret < 2000'
            else:
                ret = 'ret < 1000'

            clipped_traj = traj[:600]
            df_per_prim.append(pd.DataFrame({
                'model_traj': [key] * len(clipped_traj),
                'time':       list(range(len(clipped_traj))),
                'height':       clipped_traj,
                'ret':         [ret] * len(clipped_traj)
            }))
        df_per_prim = pd.concat(df_per_prim)
        all_df.append(df_per_prim)
    all_df = pd.concat(all_df)

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5),tight_layout=True)
    sns.lineplot(
        data= all_df,
        hue= 'model_traj',
        x= 'time',
        y= 'height',
        #style= 'ret',
        palette= ['darkorange', 'deepskyblue', 'limegreen', 'deeppink'],
        markers=False,
        ax= ax,
    )
    ax.legend().set_title('')
    ax.set_ylim([0.9, 1.8])

    ax.set_ylabel('Height', fontsize=12)
    ax.set_xlabel('time step', fontsize=13)

    plt.show()


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-10/',
        'best',
        1,
    )
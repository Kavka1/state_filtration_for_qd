from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy


def main(path: str, remark: str, chosen_primitive: List[int]) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env_config   = config['env_config']
    
    num_primitive = config['num_primitive']
    env = call_env(env_config)
    env.apply_missing_obs = False            # output the clipped observation
    
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


    height_traj_dict = dict()
    for i, policy in enumerate(all_Pis):
        traj_seq = []
        for j in range(num_traj_per_prim):
            height_traj = []
            obs = env.reset()
            done = False
            while not done:
                action = policy.act(torch.from_numpy(obs).float(), False).detach().numpy()
                obs_, r, done, info = env.step(action)
                height_traj.append(obs[0])      # must guarantee obs[0] is the value at global y-coordinate
                obs = obs_

            #if len(height_traj) > min_traj_len:
            traj_seq.append(height_traj)
        height_traj_dict.update({f'model_{i}': traj_seq})

    all_df = []
    for key in list(height_traj_dict.keys()):
        df_per_prim = []
        for j in range(len(height_traj_dict[key])):
            traj = height_traj_dict[key][j]
            clipped_traj = traj
            df_per_prim.append(pd.DataFrame({
                'model_traj': [key + f'_traj_{j}'] * len(clipped_traj),
                'time':       list(range(len(clipped_traj))),
                'height':       clipped_traj
            }))
        df_per_prim = pd.concat(df_per_prim)
        all_df.append(df_per_prim)

    fig, axs = plt.subplots(nrows=2,ncols=num_primitive//2,figsize=(18,5),tight_layout=True,sharex=True,sharey=True)
    for i, ax in enumerate(axs.flat):
        sns.lineplot(
            data= all_df[i],
            hue= 'model_traj',
            x= 'time',
            y= 'height',
            ax= ax,
        )
        ax.legend().set_title('')

    plt.show()


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/HalfCheetah-missing_leg_1-10/',
        'best',
        3,
    )
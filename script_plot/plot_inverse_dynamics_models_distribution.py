from typing import List, Tuple, Dict
import numpy as np
import torch
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


from state_filtration_for_qd.model.dynamics import DiagGaussianIDM
from state_filtration_for_qd.env.common import call_env


def main(path: str, remark: str, num_obs: int, title: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env_config   = config['env_config']
    
    num_primitive = config['num_primitive']
    env = call_env(env_config)
    env.apply_missing_obs = True            # output the clipped observation
    all_IDMs = []
    for k in range(num_primitive):
        model = DiagGaussianIDM(
            model_config['filtrated_o_dim'],
            model_config['a_dim'],
            model_config['idm_hidden_layers'],
            model_config['idm_logstd_min'],
            model_config['idm_logstd_max']
        )
        model.load_state_dict(torch.load(path + f'model/inverse_model_{k}_{remark}', map_location='cpu'))
        all_IDMs.append(model)

    # collect plenty observation data
    all_filtrated_obs       = []
    all_filtrated_next_obs  = []
    obs = env.reset()
    while len(all_filtrated_obs) < num_obs:
        a = env.action_space.sample()
        obs_, r, done, info = env.step(a)
        all_filtrated_obs.append(obs.tolist())
        all_filtrated_next_obs.append(obs_.tolist())
        if done:
            obs = env.reset()
        obs = obs_

    # inference the actions of all IDMs
    all_obs_tensor = torch.as_tensor(all_filtrated_obs).float()
    all_next_obs_tensor = torch.as_tensor(all_filtrated_next_obs).float()
    all_pred_action_mean = {f'IDM {k}': all_IDMs[k](all_obs_tensor, all_next_obs_tensor).mean.detach().numpy() for k in range(num_primitive)}

    # Plot
    sns.set_style('whitegrid') 
    fig, axs = plt.subplots(nrows=1, ncols=env.action_space.shape[0]-1, tight_layout=True, )#figsize=(10, 5),)
    # Process dataframe
    all_action_mean_df = []
    for k in range(num_primitive):
        temp_dict = {'Inverse Model': [f'model {k}'] * len(all_obs_tensor)}
        temp_dict.update({ f'Action Dim {i}': all_pred_action_mean[f'IDM {k}'][:,i] for i in range(env.action_space.shape[0])})
        all_action_mean_df.append(pd.DataFrame(temp_dict))
    all_action_mean_df = pd.concat(all_action_mean_df)
    
    for i, ax in enumerate(axs):
        sns.scatterplot(
            data= all_action_mean_df,
            x = f'Action Dim {i}',
            y = f'Action Dim {i+1}',
            hue = 'Inverse Model',
            style = 'Inverse Model',
            ax = ax
        )

        ax.set_xlabel(f'Action Dimension {i}', fontsize=11)
        ax.set_ylabel(f'Action Dimension {i+1}', fontsize=11)
        if i == len(axs)//2:
            ax.set_title(title)
            ax.legend().set_title('')
        else:
            ax.legend().remove()
    
    plt.show()


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_joint_hip_ankle-10/',
        'final',
        150,
        'IDM prediction - Ant with missing obs in joint hip & ankle '
    )
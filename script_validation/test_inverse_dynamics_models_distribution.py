from typing import List, Tuple, Dict
import numpy as np
import torch
import pandas as pd
import yaml

from state_filtration_for_qd.model.dynamics import DiagGaussianIDM
from state_filtration_for_qd.env.common import call_env


def main(path: str, remark: str, num_obs: int, csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env_config   = config['env_config']
    
    num_primitive = config['num_primitive']
    env = call_env(env_config)
    all_IDMs = []
    for k in range(num_primitive):
        model = DiagGaussianIDM(
            model_config['filtrated_o_dim'],
            model_config['a_dim'],
            model_config['idm_hidden_layers'],
            model_config['model_logstd_min'],
            model_config['model_logstd_max']
        )
        model.load_state_dict(torch.load(path + f'model/inverse_model_{k}_{remark}', map_location='cpu'))
        all_IDMs.append(model)

    # collect plenty observation data
    all_filtrated_obs = []
    obs = env.reset()
    while len(all_filtrated_obs) < num_obs:
        all_filtrated_obs.append(obs.tolist())
        a = env.action_space.sample()
        obs, r, done, info = env.step()
        if done:
            obs = env.reset()

    # inference the actions of all IDMs
    all_obs_tensor = torch.as_tensor(all_filtrated_obs).type(torch.float64)
    all_pred_action_mean = {
        f'IDM {k}': all_IDMs[k](all_obs_tensor).mean.detach().tolist() for k in range(num_primitive)
    }

    dc = all_pred_action_mean.update({'observation': all_filtrated_obs})
    df = pd.DataFrame(dc)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/',
        'final',
        100,
        '/home/xukang/Project/state_filtration_for_qd/statistic/'
    )
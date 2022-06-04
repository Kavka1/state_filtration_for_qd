from typing import List, Dict, Tuple
import numpy as np
import torch
import pandas as pd
import yaml
from openTSNE import TSNE

from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.model.dynamics import DiagGaussianIDM


def main(path: str, remark: str, num_obs: int, csv_path: str, csv_remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    env_config   = config['env_config']
    
    num_primitive = config['num_primitive']
    env = call_env(env_config)
    env.apply_missing_obs = False            # output the clipped observation
    all_IDMs = []
    all_Pis  = []
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

        policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        policy.load_model(config['exp_path'] + f'models/policy_{remark}')
        all_Pis.append(policy)

    # collect plenty observation data
    all_obs                 = []
    all_filtrated_obs       = []
    all_filtrated_next_obs  = []
    obs = env.reset()
    while len(all_filtrated_obs) < num_obs:
        a = env.action_space.sample()
        obs_, r, done, info = env.step(a)
        all_obs.append(obs.tolist())
        all_filtrated_obs.append(env._process_obs(obs).tolist())
        all_filtrated_next_obs.append(env._process_obs(obs_).tolist())
        if done:
            obs = env.reset()
        obs = obs_

    # inference the actions of all IDMs
    all_obs_tensor = torch.as_tensor(all_obs).float()
    all_filtrated_obs_tensor = torch.as_tensor(all_filtrated_obs).float()
    all_filtrated_next_obs_tensor = torch.as_tensor(all_filtrated_next_obs).float()

    # train the tsne embedding for the policy output
    X = [all_Pis[k](all_obs_tensor).mean().detach().numpy() for k in range(num_primitive)]
    y = [np.array([i for _ in range(num_obs)]) for i in range(num_primitive)]
    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0).tolist()

    embedding = TSNE().fit(X)
    df = pd.DataFrame({'embedding': embedding, 'model': y})
    df.to_csv(csv_path + f'{csv_remark}-policy.csv')

    # train the tsne embedding for the idm output
    X_idm = [all_IDMs[k](all_filtrated_obs_tensor, all_filtrated_next_obs_tensor).mean().detach().numpy() for k in range(num_primitive)]
    y_idm = [np.array([i for _ in range(num_obs)]) for i in range(num_primitive)]
    X_idm = np.concatenate(X_idm, 0)
    y_idm = np.concatenate(y_idm, 0).tolist()

    embedding_idm = TSNE().fit(X_idm)
    df_idm = pd.DataFrame({'embedding': embedding_idm, 'model': y_idm})
    df_idm.to_csv(csv_path + f'{csv_remark}-idm.csv')


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-10/',
        'final',
        5000,
        '/home/xukang/Project/state_filtration_for_qd/statistic/tsne/',
        'walker-missing_leg_1-10'
    )
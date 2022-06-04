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
    
    num_primitive = 5# config['num_primitive']
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
        policy.load_model(path + f'model/policy_{k}_{remark}')
        all_Pis.append(policy)

    # collect plenty observation data
    all_obs                 = [[] for _ in range(num_primitive)]
    all_filtrated_obs       = []
    all_filtrated_next_obs  = []
    all_policy_action_mean  = [[] for _ in range(num_primitive)]
    all_idm_action_mean     = [[] for _ in range(num_primitive)]
    for i, policy in enumerate(all_Pis):
        obs = env.reset()
        step = 0
        while step < num_obs:
            a = policy.act(torch.from_numpy(obs).float(), False).detach().numpy()
            obs_, r, done, info = env.step(a)
            
            filtrated_obs = env._process_obs(obs)
            filtrated_next_obs = env._process_obs(obs_)
            a_idm = all_IDMs[i](torch.from_numpy(filtrated_obs).float(), torch.from_numpy(filtrated_next_obs).float()).mean.detach().numpy()

            all_obs[i].append(obs)
            all_policy_action_mean[i].append(a)
            all_idm_action_mean[i].append(a_idm)

            if done:
                obs = env.reset()
            obs = obs_
            step += 1

    # train the tsne embedding for the policy output
    y = [np.array([i for _ in range(num_obs)]) for i in range(num_primitive)]
    X = np.concatenate(all_obs, 0)
    y = np.concatenate(y, 0).tolist()
    embedding = TSNE().fit(X)
    df = pd.DataFrame({'embedding': [list(embedding[i]) for i in range(len(embedding))], 'model': y})
    df.to_csv(csv_path + f'{csv_remark}-obs.csv')
    print("t-SNE of visited obs saved to csv")

    # train the tsne embedding for the policy output
    y = [np.array([i for _ in range(num_obs)]) for i in range(num_primitive)]
    X = np.concatenate(all_policy_action_mean, 0)
    y = np.concatenate(y, 0).tolist()
    embedding = TSNE().fit(X)
    df = pd.DataFrame({'embedding': [list(embedding[i]) for i in range(len(embedding))], 'model': y})
    df.to_csv(csv_path + f'{csv_remark}-policy.csv')
    print("t-SNE of policy output saved to csv")

    # train the tsne embedding for the idm output
    y_idm = [np.array([i for _ in range(num_obs)]) for i in range(num_primitive)]
    X_idm = np.concatenate(all_idm_action_mean, 0)
    y_idm = np.concatenate(y_idm, 0).tolist()
    embedding_idm = TSNE().fit(X_idm)
    df_idm = pd.DataFrame({'embedding': [list(embedding_idm[i]) for i in range(len(embedding_idm))], 'model': y_idm})
    df_idm.to_csv(csv_path + f'{csv_remark}-idm.csv')
    print("t-SNE of IDM output saved to csv")


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_leg_1-10/',
        'final',
        500,
        '/home/xukang/Project/state_filtration_for_qd/statistic/tsne/',
        'walker-missing_leg_1-10'
    )
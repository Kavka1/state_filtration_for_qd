from datetime import datetime
from typing import Dict, List, Tuple, Callable
import gym
import datetime
from importlib_metadata import os, sys
import numpy as np
import torch
import torch.nn as nn
import yaml


def hard_update(source_model: nn.Module, target_model: nn.Module) -> None:
    source_model.load_state_dict(target_model.state_dict())


def soft_update(source_model: nn.Module, target_model: nn.Module, tau: float) -> None:
    assert tau < 0.5, "Value of tau must be less than 0.5"
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def confirm_path_exist(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def seed_all(seed: int, test_env: gym.Env) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_env.seed(seed)


def gae_estimator(rewards: List[float], values: List[np.float32], gamma: float, lamda: float) -> Tuple[List, List]:
    ret_seq, adv_seq = [], []
    prev_ret, prev_adv, prev_value = 0., 0., 0.
    length = len(rewards)
    for i in reversed(range(length)):
        ret = rewards[i] + gamma * prev_ret
        delta = rewards[i] + gamma * prev_value - values[i]
        adv = delta + gamma * lamda * prev_adv

        ret_seq.insert(0, ret)
        adv_seq.insert(0, adv)
        
        prev_ret = ret
        prev_value = values[i]
        prev_adv = adv

    return ret_seq, adv_seq


def make_exp_path(config: Dict, exp_name: str) -> None:
    # concat experiment name
    exp_file = "" if exp_name is '' else f'{exp_name}-'
    # concat env name
    exp_file += f"{config['env_name']}"
    # concat missing observation info
    for key, missing_item in config['missing_obs_info'].items():
        if len(missing_item) == 0:
            continue
        concat_missing_info = ''
        for item in missing_item:
            concat_missing_info += f'_{item}'
        exp_file += f'-{key}' + f'{concat_missing_info}' 
    # concat seed
    exp_file += f"-{config['seed']}"

    exp_path = config['result_path'] + exp_file
    while os.path.exists(exp_path):
        exp_path += '_*'
    config.update({'exp_path': exp_path + '/'})
    confirm_path_exist(config['exp_path'])
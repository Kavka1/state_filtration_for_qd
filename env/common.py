from typing import Dict, List, Tuple
import numpy as np
import gym

from state_filtration_for_qd.env.walker import Missing_Info_Walker
from state_filtration_for_qd.env.ant import Missing_Info_Ant
from state_filtration_for_qd.env.hopper import Missing_Info_Hopper
from state_filtration_for_qd.env.swimmer import Missing_Info_Swimmer
from state_filtration_for_qd.env.halfcheetah import Missing_Info_HalfCheetah


NAME2Env = {
    'Hopper': Missing_Info_Hopper,
    'Walker': Missing_Info_Walker,
    'HalfCheetah': Missing_Info_HalfCheetah,
    'Ant': Missing_Info_Ant,
    'Swimmer': Missing_Info_Swimmer
}


def call_env(env_config: Dict) -> gym.Env:
    name = env_config['env_name']
    missing_obs_info = env_config['missing_obs_info']

    if name in list(NAME2Env.keys()):
        env = NAME2Env[name](missing_obs_info)
    else:
        raise ValueError(f"Invalid env name {name}")
    
    return env
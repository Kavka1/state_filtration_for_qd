from typing import Dict, List, Tuple
import numpy as np
import gym

from state_filtration_for_qd.env.walker import Missing_Info_Walker
from state_filtration_for_qd.env.ant import Missing_Info_Ant
from state_filtration_for_qd.env.hopper import Missing_Info_Hopper
from state_filtration_for_qd.env.swimmer import Missing_Info_Swimmer
from state_filtration_for_qd.env.dmc_quadruped import Missing_Info_Quadruped_Walk
from state_filtration_for_qd.env.minitaur import Missing_Info_Minitaur
from state_filtration_for_qd.env.halfcheetah import Missing_Info_HalfCheetah
from state_filtration_for_qd.env.hopper_broken_leg import Broken_Leg_Hopper
from state_filtration_for_qd.env.walker_broken_leg import Broken_Leg_Walker
from state_filtration_for_qd.env.walker_disturb_dynamics import Disturb_Dynamics_Walker
from state_filtration_for_qd.env.hopper_disturb_dynamics import Disturb_Dynamics_Hopper


NAME2Env = {
    'Hopper': Missing_Info_Hopper,
    'Walker': Missing_Info_Walker,
    'HalfCheetah': Missing_Info_HalfCheetah,
    'Ant': Missing_Info_Ant,
    'Swimmer': Missing_Info_Swimmer,
    'Minitaur': Missing_Info_Minitaur,
    'Quadruped': Missing_Info_Quadruped_Walk
}

NAME2BROKEN_ENV = {
    'Hopper': Broken_Leg_Hopper,
    'Walker': Broken_Leg_Walker
}

NAME2DYNAMICS_ENV = {
    'Hopper': Disturb_Dynamics_Hopper,
    'Walker': Disturb_Dynamics_Walker
}


def call_env(env_config: Dict, is_render: bool = False) -> gym.Env:
    name = env_config['env_name']
    missing_obs_info = env_config['missing_obs_info']
    if name in list(NAME2Env.keys()):
        if name == "Minitaur":
            return NAME2Env[name](missing_obs_info=missing_obs_info, render=is_render)
        else:
            return NAME2Env[name](missing_obs_info=missing_obs_info)
    else:
        raise ValueError(f"Invalid env name {name}")


def call_broken_leg_env(env_config: Dict) -> gym.Env:
    name = env_config['env_name']
    args = env_config['broken_leg_info']
    if name in list(NAME2BROKEN_ENV.keys()):
        return NAME2BROKEN_ENV[name](**args)
    else:
        raise ValueError(f"Invalid broken env name {name}")


def call_disturb_dynamics_env(env_config: Dict) -> gym.Env:
    name = env_config['env_name']
    args = env_config['dynamics_info']
    if name in list(NAME2BROKEN_ENV.keys()):
        return NAME2DYNAMICS_ENV[name](**args)
    else:
        raise ValueError(f"Invalid dynamics disturbed env name {name}")
    
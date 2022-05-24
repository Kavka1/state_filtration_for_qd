from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.noise_obs_wrapper import Noise_Obs_Wrapper
from state_filtration_for_qd.env.common import call_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        self.model.load_model(model_path)
        self.num_episode = num_episode

    def set_env(self, env_config: Dict, noise_type: str, noise_scale: float, noise_index: List[int])-> None:
        self.env = Noise_Obs_Wrapper(call_env(env_config), noise_type, noise_scale, noise_index)

    def rollout(self) -> float:
        total_reward = 0
        for _ in range(self.num_episode):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.model.act(
                    torch.from_numpy(obs).float(),
                    False
                )
                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode



def main(path: str, remark: str, noise_index: List[int], csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['num_primitive']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], 20))

    noise_range = [round(j * 0.1 + 0, 2) for j in range(16)]
    score_dict = {'noise scale': noise_range}
    score_dict.update({f'primitive {k}': [] for k in range(num_primitive)})


    for noise_scale in noise_range:
        remotes = [worker.set_env.remote(config['env_config'], 'Gaussian', noise_scale, noise_index) for worker in all_workers]
        ray.get(remotes)

        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)

        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = [all_primitive_scores[j] for j in range(num_primitive)]
        
        print(f'\nnoise scale {noise_scale}')
        for k in range(num_primitive):
            print(f"    Primitive {k}: {all_primitive_scores[k]}")
        print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

        for k in range(num_primitive):
            score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main(
        path='/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Ant-missing_coord_2_3_4_5-10/',
        remark='best',
        noise_index=[1,2,3,4,5,6,7,8,9,10,11,12],
        csv_path='/home/xukang/Project/state_filtration_for_qd/statistic/Ant_coord_2_3_4_5-10.csv'
    )
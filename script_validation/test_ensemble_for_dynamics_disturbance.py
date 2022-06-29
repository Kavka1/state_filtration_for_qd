from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import ray
import torch.nn as nn
import yaml
import pandas as pd

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_disturb_dynamics_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, num_episode: int) -> None:
        self.model = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            model_config['policy_activation']
        )
        self.model.load_model(model_path)
        self.num_episode = num_episode

    def set_env(self, env_config: Dict)-> None:
        self.env = call_disturb_dynamics_env(env_config)

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



def main(path: str, remark: str, env_config: Dict, disturbed_param: List[str], csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = 1 #config['num_primitive']
    all_workers = []
    for k in range(num_primitive):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], 20))

    parameter_scale_range = [round(j * 0.1 + 0.5, 2) for j in range(20)]
    score_dict = {'param scale': parameter_scale_range}
    score_dict.update({f'primitive {k}': [] for k in range(num_primitive)})

    for param_scale in parameter_scale_range:
        if 'mass' in disturbed_param:
            env_config['dynamics_info'].update({
                'leg_mass_scale':      param_scale,
            })
        if 'fric' in disturbed_param:
            env_config['dynamics_info'].update({
                'ankle_friction_scale':  param_scale
            })

        remotes = [worker.set_env.remote(env_config) for worker in all_workers]
        ray.get(remotes)
        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)
        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = [all_primitive_scores[j] for j in range(num_primitive)]
        
        print(f'\param scale {param_scale}')
        for k in range(num_primitive):
            print(f"    Primitive {k}: {all_primitive_scores[k]}")
        print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

        for k in range(num_primitive):
            score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    for env in [
        #'Hopper',
        #'Walker'
        'Ant'
    ]:
        if env == 'Hopper':
            path_mark = 'missing_leg_1'
        elif env == 'Walker':
            path_mark = 'missing_leg_1'
        elif env == 'Ant':
            path_mark = 'missing_leg_1_2_3_4'
            
        for seed in [
            #10, 20, 30, 40, 50
            60, 70, 80
        ]:
            for disturb_param in [['mass'],['fric']]:

                main(
                    path=f'/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/{env}-{path_mark}-{seed}/',
                    remark='best',
                    env_config={
                        'env_name': env,
                        'dynamics_info': {
                            'leg_mass_scale': 1,
                            'ankle_friction_scale': 1,
                        }
                    },
                    disturbed_param= disturb_param,
                    csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_dynamics_{disturb_param[0]}-{seed}.csv'
                )
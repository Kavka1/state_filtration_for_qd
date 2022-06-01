from typing import List, Dict, Tuple
import numpy as np
import torch
import ray
import pandas as pd
import yaml

from state_filtration_for_qd.model.latent_policy import Latent_DiagGaussianPolicy
from state_filtration_for_qd.env.common import call_disturb_dynamics_env


@ray.remote
class Worker(object):
    def __init__(self, model_path: str, model_config: Dict, num_episode: int, z: int) -> None:
        self.model = Latent_DiagGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['z_dim'],
            model_config['policy_hidden_layers'],
            model_config['policy_logstd_min'],
            model_config['policy_logstd_max']
        )
        self.model.load_model(model_path)
        self.z = z
        self.z_one_hot = np.zeros([model_config['z_dim'],]).astype(np.float64)
        self.z_one_hot[z] = 1
        
        self.num_episode = num_episode

    def set_env(self, env_config: Dict)-> None:
        self.env = call_disturb_dynamics_env(env_config)

    def rollout(self) -> float:
        total_reward = 0
        for _ in range(self.num_episode):
            done = False
            obs = self.env.reset()
            while not done:
                obs_z = np.concatenate([obs, self.z_one_hot], -1)
                a = self.model.act(
                    torch.from_numpy(obs_z).float(),
                    False
                ).detach().numpy()
                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode



def main(path: str, remark: str, env_config: Dict, disturbed_param: List[str], csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    z_dim = config['model_config']['z_dim']
    all_workers = []
    model_path = path + f'model/policy_{remark}'
    for k in range(z_dim):
        all_workers.append(Worker.remote(model_path, config['model_config'], 20, k))

    parameter_scale_range = [round(j * 0.1 + 0.5, 2) for j in range(20)]
    score_dict = {'param scale': parameter_scale_range}
    score_dict.update({f'primitive {k}': [] for k in range(z_dim)})

    for param_scale in parameter_scale_range:
        if 'mass' in disturbed_param:
            env_config['dynamics_info'].update({
                'foot_mass_scale':      param_scale,
            })
        if 'fric' in disturbed_param:
            env_config['dynamics_info'].update({
                'foot_friction_scale':  param_scale
            })

        remotes = [worker.set_env.remote(env_config) for worker in all_workers]
        ray.get(remotes)
        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)
        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = [all_primitive_scores[j] for j in range(z_dim)]
        
        print(f'\param scale {param_scale}')
        for k in range(z_dim):
            print(f"    Primitive {k}: {all_primitive_scores[k]}")
        print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

        for k in range(z_dim):
            score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)



if __name__ == '__main__':
    env = 'Hopper'
    disturb_param = ['mass']
    for seed in [10, 20, 30]:
        main(
            path=f'/home/xukang/Project/state_filtration_for_qd/results_for_diayn/r_ex-{env}-{seed}/',
            remark='best',
            env_config={
                'env_name': env,
                'dynamics_info': {
                    'foot_mass_scale': 1,
                    'foot_friction_scale': 1
                }
            },
            disturbed_param= disturb_param,
            csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{env}_dynamics_{disturb_param[0]}-{seed}.csv'
        )
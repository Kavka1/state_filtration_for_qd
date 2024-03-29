from typing import List, Dict, Tuple
import numpy as np
import torch
import ray
import pandas as pd
import yaml

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.env.noise_obs_wrapper import Noise_Obs_Wrapper


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
                ).detach().numpy()
                obs, r, done, info = self.env.step(a)
                total_reward += r
        return total_reward / self.num_episode



def main(path: str, remark: str, noise_index: List[int], csv_path: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    z_dim = config['model_config']['z_dim']
    all_workers = []
    for k in range(z_dim):
        model_path = path + f'model/policy_{k}_{remark}'
        all_workers.append(Worker.remote(model_path, config['model_config'], 20))

    noise_range = [round(j * 0.1 + 0, 2) for j in range(16)]
    score_dict = {'noise scale': noise_range}
    score_dict.update({f'primitive {k}': [] for k in range(z_dim)})

    for noise_scale in noise_range:
        remotes = [worker.set_env.remote(config['env_config'], 'Gaussian', noise_scale, noise_index) for worker in all_workers]
        ray.get(remotes)

        rollout_remote = [worker.rollout.remote() for worker in all_workers]
        all_primitive_scores = ray.get(rollout_remote)

        best_primitive_index = np.argmax(all_primitive_scores)
        all_primitive_scores = [all_primitive_scores[j] for j in range(z_dim)]
        
        print(f'\nnoise scale {noise_scale}')
        for k in range(z_dim):
            print(f"    Primitive {k}: {all_primitive_scores[k]}")
        print(f"    Best primitive: {best_primitive_index}  -  {all_primitive_scores[best_primitive_index]}")

        for k in range(z_dim):
            score_dict[f'primitive {k}'].append(all_primitive_scores[k].item())
        
    score_df = pd.DataFrame(score_dict)
    score_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    for env, noise_idx in zip([
        'Hopper',
        #'Walker',
        #'Ant'
    ],[
        [2,3,4,8,9,10],
        #[2,3,4,11,12,13],
        #list(range(1,13))
    ]):
        if env == 'Hopper':
            path_mark = 'missing_leg_1'
            csv_mark = 'leg_1'
        elif env == 'Walker':
            path_mark = 'missing_leg_1'
            csv_mark = 'leg_1'
        else:
            path_mark = 'missing_leg_1_2_3_4'
            csv_mark = 'coord_2_3_4_5'

        for seed in [10, 20, 30, 40, 50]:
            main(
                path=f'/home/xukang/Project/state_filtration_for_qd/results_for_diayn/ppo_ensemble-r_ex-{env}-{seed}/',
                remark='best',
                noise_index=noise_idx,
                csv_path=f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn_ppo/ppo_{env}_{csv_mark}-{seed}.csv'
            )
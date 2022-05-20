from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import ray

from state_filtration_for_qd.model.latent_policy import Latent_FixStdGaussianPolicy
from state_filtration_for_qd.model.value import VFunction
from state_filtration_for_qd.model.discriminator import DeltaSA_Z_Discrete_Discriminator
from state_filtration_for_qd.utils import gae_estimator
from state_filtration_for_qd.env.common import call_env


@ray.remote
class RayWorker(object):
    def __init__(
        self,
        worker_id: int, 
        model_config: Dict,
        reward_tradeoff: float,
        env_config: str,
        seed: int, 
        gamma: float,
        lamda: float,
        rollout_episodes: int,
    ) -> None:
        self.id = worker_id
        self.tradeoff = reward_tradeoff
        self.z_dim = model_config['z_dim']
        self.p_z = np.full(self.z_dim, 1 / self.z_dim)

        self.policy = Latent_FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['z_dim'],
            model_config['policy_hidden_layers'],
            model_config['policy_ac_std']
        )
        self.value = VFunction(
            model_config['o_dim'] + model_config['z_dim'],
            model_config['value_hidden_layers']
        )
        self.discriminator = DeltaSA_Z_Discrete_Discriminator(
            model_config['filtrated_o_dim'],
            model_config['a_dim'],
            model_config['z_dim'],
            model_config['disc_hidden_layers']
        )

        self.env = call_env(env_config)
        self.env.seed(seed)
        self.obs_filter = self.env._process_obs

        self.gamma, self.lamda = gamma, lamda
        self.rollout_episodes = rollout_episodes
        self.action_bound = self.env.action_bound

    def rollout(
        self,
        policy_state_dict: nn.ModuleDict,
        value_state_dict: nn.ModuleDict,
        disc_state_dict: nn.ModuleDict
    ) -> Dict:
        self.policy.load_state_dict(policy_state_dict)
        self.value.load_state_dict(value_state_dict)
        self.discriminator.load_state_dict(disc_state_dict)

        all_obs_z_seq = []
        all_filtered_obs_seq = []
        all_filtered_next_obs_seq = []
        all_a_seq = [] 
        all_logprob_seq = []
        all_ret_seq = [] 
        all_adv_seq = []
        all_z_seq = []

        all_r_ex_seq = []
        all_r_in_seq = []

        total_steps, cumulative_ex_r , cumulative_in_r = 0, 0, 0

        for episode in range(self.rollout_episodes):
            obs_z_seq, filtered_obs_seq, filtered_next_obs_seq, a_seq, z_seq, r_seq, logprob_seq, value_seq = [], [], [], [], [], [], []
            r_ex_seq, r_in_seq = [], []

            obs = self.env.reset()
            done = False

            z = np.random.choice(self.z_dim, p=self.p_z)
            one_hot_z = np.zeros(self.z_dim).astype(np.float64)
            one_hot_z[z] = 1

            while not done:
                obs_z = np.concatenate([obs, one_hot_z], axis=-1)
                obs_z_tensor = torch.from_numpy(obs_z).float()

                a, log_prob, dist = self.policy(obs_z_tensor)
                value = self.value(obs_z_tensor).squeeze(-1).detach().numpy()
                log_prob = log_prob.detach().numpy()
                a = a.detach().numpy()
                
                clipped_a = np.clip(a, -self.action_bound, self.action_bound)
                obs_, r_ex, done, info = self.env.step(clipped_a)

                # inference the logP(z|s'-s, a) and denote as intrinsic reward
                # filtrate the observation
                obs_filtered, n_obs_filtered = self.obs_filter(obs), self.obs_filter(obs_)
                logprob_z = self.discriminator.inference(
                    torch.from_numpy(obs_filtered).float(),
                    torch.from_numpy(n_obs_filtered).float(),
                    torch.from_numpy(a),
                    torch.from_numpy(z).type(torch.int32)
                )
                r_in = logprob_z.tolist()[0]
                # combine reward
                r = r_ex + self.tradeoff * r_in
                
                obs_z_seq.append(obs_z)
                filtered_obs_seq.append(obs_filtered)
                filtered_next_obs_seq.append(n_obs_filtered)
                a_seq.append(a)
                z_seq.append(z)
                logprob_seq.append(log_prob)
                value_seq.append(value)

                r_seq.append(r)
                r_ex_seq.append(r_ex)
                r_in_seq.append(r_in)

                cumulative_ex_r += r_ex
                cumulative_in_r += r_in

                total_steps += 1
                obs = obs_

            ret_seq, adv_seq = gae_estimator(r_seq, value_seq, self.gamma, self.lamda)
            
            all_obs_z_seq += obs_z_seq
            all_filtered_obs_seq += filtered_obs_seq
            all_filtered_next_obs_seq += filtered_next_obs_seq
            all_a_seq += a_seq
            all_logprob_seq += logprob_seq
            all_ret_seq += ret_seq
            all_adv_seq += adv_seq

            all_z_seq += z_seq
            all_r_ex_seq += r_ex_seq
            all_r_in_seq += r_in_seq
        
        return {
            'steps': total_steps,
            'episode_r_ex': cumulative_ex_r / self.rollout_episodes,
            'step_r_in': cumulative_in_r / total_steps,

            'obs_z': np.stack(all_obs_z_seq, 0),
            'filtered_obs': np.stack(all_filtered_obs_seq, 0),
            'filtered_next_obs': np.stack(all_filtered_next_obs_seq, 0),
            'a': np.stack(all_a_seq, 0),
            'z': np.stack(all_z_seq, 0),
            'logprob': np.stack(all_logprob_seq, 0),
            'ret': np.array(all_ret_seq, dtype=np.float32)[:, np.newaxis],
            'adv': np.array(all_adv_seq, dtype=np.float32)[:, np.newaxis]
        }


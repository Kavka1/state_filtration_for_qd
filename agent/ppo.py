from math import gamma
from typing import Callable, Dict, List, Tuple
import torch
import ray
from copy import copy, deepcopy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from state_filtration_for_qd.utils import confirm_path_exist, gae_estimator
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.buffer import Dataset
from state_filtration_for_qd.model.latent_policy import Latent_FixStdGaussianPolicy
from state_filtration_for_qd.model.discriminator import DeltaSA_Z_Discrete_Discriminator
from state_filtration_for_qd.model.value import VFunction



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
            model_config['action_std']
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
            obs_z_seq = []
            filtered_obs_seq = [] 
            filtered_next_obs_seq = [] 
            a_seq = []
            z_seq = []
            r_seq = []
            logprob_seq = []
            value_seq = []
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
                    torch.from_numpy(np.array(z)).type(torch.int32)
                )
                r_in = float(logprob_z)
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
            'z': np.array(all_z_seq, dtype=np.int64)[:, np.newaxis],
            'logprob': np.stack(all_logprob_seq, 0),
            'ret': np.array(all_ret_seq, dtype=np.float32)[:, np.newaxis],
            'adv': np.array(all_adv_seq, dtype=np.float32)[:, np.newaxis]
        }




class PPOAgent(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_config = config['model_config']

        self.tradeoff = config['reward_tradeoff']
        self.lr = config['lr']
        self.gamma = config['gamma']
        self.lamda = config['lamda']
        self.ratio_clip = config['ratio_clip']
        self.temperature_coef = config['temperature_coef']
        self.num_epoch = config['num_epoch']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.device = torch.device(config['device'])
        self.exp_path = config['exp_path']

        # Model Definition
        self.policy = Latent_FixStdGaussianPolicy(
            self.model_config['o_dim'],
            self.model_config['a_dim'],
            self.model_config['z_dim'],
            self.model_config['policy_hidden_layers'],
            self.model_config['action_std']
        ).to(self.device)
        self.value = VFunction(
            self.model_config['o_dim'] + self.model_config['z_dim'],
            self.model_config['value_hidden_layers']
        ).to(self.device)
        self.discriminator = DeltaSA_Z_Discrete_Discriminator(
            self.model_config['filtrated_o_dim'],
            self.model_config['a_dim'],
            self.model_config['z_dim'],
            self.model_config['disc_hidden_layers'],
        ).to(self.device)
        # Optimizer Definition
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.lr)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.workers = []
        self.total_steps, self.total_episodes = 0, 0
        
    def _init_workers(self, env_config: str, initial_seed: int, rollout_episodes: int, seed_increment: bool = True) -> None:
        self.num_worker_rollout = rollout_episodes
        for i in range(self.num_workers):
            self.workers.append(
                RayWorker.remote(
                    worker_id = i, 
                    model_config = self.model_config,
                    reward_tradeoff = self.tradeoff,
                    env_config = env_config,
                    seed = initial_seed + i * seed_increment, 
                    gamma = self.gamma,
                    lamda = self.lamda,
                    rollout_episodes = rollout_episodes,
                )
            )

    def choose_action(self, obs: np.array, one_hot_z: np.array, with_noise: bool = True) -> np.array:
        obs_z = np.concatenate([obs, one_hot_z], -1)
        obs_z = torch.from_numpy(obs_z).to(self.device).float()
        action = self.policy.act(obs_z, with_noise).detach().numpy()
        return action

    def roll_update(self) -> Tuple[float, List, float, float]:
        policy_state_dict_remote = ray.put(deepcopy(self.policy).to(torch.device('cpu')).state_dict())
        value_state_dict_remote = ray.put(deepcopy(self.value).to(torch.device('cpu')).state_dict())
        disc_state_dict_remote = ray.put(deepcopy(self.discriminator).to(torch.device('cpu')).state_dict())
        
        rollout_remote = [worker.rollout.remote(
            policy_state_dict_remote,
            value_state_dict_remote,
            disc_state_dict_remote
        ) for i, worker in enumerate(self.workers)]
        results = ray.get(rollout_remote)

        return_ex, bonus_in, worker_returns = 0, 0, []
        log_loss_pi, log_loss_v, log_loss_disc, update_count = 0, 0, 0, 0
        
        data_buffer = {
            'obs_z': [], 
            'filtered_obs': [], 
            'filtered_next_obs': [],
            'a': [], 
            'z': [],
            'logprob': [],
            'ret': [], 
            'adv': []
        }
        for item in results:
            self.total_steps += item['steps']
            self.total_episodes += self.num_worker_rollout * self.num_workers

            return_ex += item['episode_r_ex']
            bonus_in += item['step_r_in']
            worker_returns.append(item['episode_r_ex'])

            data_buffer['obs_z'].append(item['obs_z'])
            data_buffer['filtered_obs'].append(item['filtered_obs'])
            data_buffer['filtered_next_obs'].append(item['filtered_next_obs'])
            data_buffer['a'].append(item['a'])
            data_buffer['z'].append(item['z'])
            data_buffer['logprob'].append(item['logprob'])
            data_buffer['ret'].append(item['ret'])
            data_buffer['adv'].append(item['adv'])
        
        return_ex /= self.num_workers
        bonus_in /= self.num_workers
        
        for key in list(data_buffer.keys()):
            data_buffer[key] = torch.from_numpy(np.concatenate(data_buffer[key], 0)).float().to(self.device)
        all_batch = Dataset(data_buffer)

        for i in range(self.num_epoch):
            for batch in all_batch.iterate_once(self.batch_size):
                obs_z, a, logprob = batch['obs_z'], batch['a'], batch['logprob']
                ret, adv =  batch['ret'], batch['adv']
                z, filtered_obs, filtered_next_obs = batch['z'], batch['filtered_obs'], batch['filtered_next_obs']

                loss_policy = self.train_policy(obs_z, a, logprob, adv)
                loss_value = self.train_value(obs_z, ret)
                loss_disc = self.train_discriminator(filtered_obs, filtered_next_obs, a, z)

                log_loss_pi += loss_policy
                log_loss_v += loss_value
                log_loss_disc += loss_disc

                update_count += 1

        return {
            'return_ex': return_ex,
            'bonus_in': bonus_in,
            'loss_policy': log_loss_pi / update_count,
            'loss_value': log_loss_v / update_count,
            'loss_disc': log_loss_disc / update_count
        }
        
    def train_policy(self, obs_z: torch.tensor, a: torch.tensor, logprob: torch.tensor, adv: torch.tensor) -> float:
        if len(adv) != 1:   # length is 1, the std will be nan
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        _, _, dist = self.policy(obs_z)
        new_logprob = dist.log_prob(a)
        entropy = dist.entropy()

        ratio = torch.exp(new_logprob.sum(-1, keepdim=True) - logprob.sum(-1, keepdim=True))
        surr1 = ratio * adv
        surr2 = torch.clip(ratio, 1-self.ratio_clip, 1+self.ratio_clip) * adv
        loss_policy = (- torch.min(surr1, surr2) - self.temperature_coef * entropy).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer_policy.step()

        return loss_policy.detach().item()

    def train_value(self, obs_z: torch.tensor, ret: torch.tensor) -> float:
        value = self.value(obs_z)
        loss_value = F.mse_loss(value, ret)

        self.optimizer_value.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer_value.step()

        return loss_value.detach().item()

    def train_discriminator(self, filtered_obs: torch.tensor, filtered_next_obs: torch.tensor, a: torch.tensor, z: torch.tensor) -> float:
        logits_z = self.discriminator(filtered_obs, filtered_next_obs, a)
        prob_z = torch.softmax(logits_z, -1)
        
        loss_disc = F.cross_entropy(prob_z, z.squeeze(-1).type(torch.int64))
        self.optimizer_disc.zero_grad()
        loss_disc.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.optimizer_disc.step()

        return loss_disc.detach().item()

    def evaluate(self, env, num_episodes: int) -> List[float]:
        rewards_across_skill = []
        for z in range(self.model_config['z_dim']):
            reward = 0
            one_hot_z = np.zeros(self.model_config['z_dim']).astype(np.float64)
            one_hot_z[z] = 1
            for i_episode in range(num_episodes):
                done = False
                obs = env.reset()
                step = 0
                while not done:
                    action = self.choose_action(obs, one_hot_z, False)
                    next_obs, r, done, info = env.step(action)
                    reward += r
                    obs = next_obs
                    step += 1
            rewards_across_skill.append(reward / num_episodes)
        return rewards_across_skill
    
    def save_policy(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'policy_{remark}'
        torch.save(self.policy.state_dict(), model_path)
        print(f"------- Policy saved to {model_path} ----------")
    
    def save_discriminator(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'disc_{remark}'
        torch.save(self.discriminator.state_dict(), model_path)
        print(f"------- Discriminator saved to {model_path} ----------")

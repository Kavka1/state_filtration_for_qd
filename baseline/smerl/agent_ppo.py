from typing import Dict, List, Tuple
import numpy as np
import torch
import ray
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from state_filtration_for_qd.utils import confirm_path_exist, gae_estimator
from state_filtration_for_qd.buffer import Dataset
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.model.discriminator import S_DiscreteZ_Discriminator
from state_filtration_for_qd.model.value import VFunction


@ray.remote
class RayWorker(object):
    def __init__(
        self,
        worker_id: int,
        model_config: Dict,
        env_config: Dict,
        seed: int,
        rollout_episode: int
    ) -> None:
        self.id = worker_id
        self.rollout_episode = rollout_episode
        
        self.policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std'],
            model_config['policy_activation']
        )
        self.env = call_env(env_config)
        self.env.seed(seed)
        self.action_bound = self.env.action_bound

    def rollout(self, policy_state_dict: nn.ModuleDict) -> Tuple:
        self.policy.load_state_dict(policy_state_dict)

        info_dict = {'step': 0, 'rewards': 0}
        trajectories = [dict() for _ in range(self.rollout_episode)]
        for i_episode in range(self.rollout_episode):
            obs_seq, a_seq, r_seq, logprob_seq = [], [], [], []
            ret = 0

            obs = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    dist = self.policy(torch.from_numpy(obs).float())
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                action, logprob = action.detach().numpy(), logprob.detach().numpy()
                cliped_a = np.clip(action, -self.action_bound, self.action_bound)
                obs_, r, done, info = self.env.step(cliped_a)
                
                # pack the trajectory
                obs_seq.append(obs)
                a_seq.append(action)
                r_seq.append(r)
                logprob_seq.append(logprob)

                info_dict['step'] += 1
                info_dict['rewards'] += r
                ret += r
                obs = obs_

            trajectories[i_episode].update({
                'obs': obs_seq,
                'action': a_seq,
                'r': r_seq,
                'logprob': logprob_seq,
                'return': ret
            })
            info_dict['rewards'] /= self.rollout_episode
        
        return (info_dict, trajectories)




class SMERL_PPO(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_config           = config['model_config']
        self.z_dim                  = config['model_config']['z_dim']
        self.num_workers            = config['num_workers']
        self.exp_path               = config['exp_path']
        self.num_worker_rollout     = config['num_worker_rollout']

        self.lr                     = config['lr']
        self.gamma                  = config['gamma']
        self.lamda                  = config['lamda']
        self.ratio_clip             = config['ratio_clip']
        self.temperature_coef       = config['temperature_coef']
        self.num_epoch              = config['num_epoch']
        self.batch_size             = config['batch_size']

        self.alpha              = config['alpha']           # the tradeoff of the intrinsic reward
        self.epsilon            = config['epsilon']         # epsilon * reward_default
        self.return_default     = config['return_default']  # the default reward trained by basic sac

        self.device = torch.device(config['device'])

        self.value = VFunction(
            self.model_config['o_dim'],
            self.model_config['value_hidden_layers']
        ).to(self.device)
        self.policies = [
            FixStdGaussianPolicy(
                self.model_config['o_dim'],
                self.model_config['a_dim'],
                self.model_config['policy_hidden_layers'],
                self.model_config['action_std'],
                self.model_config['policy_activation']
            ).to(self.device)
            for _ in range(self.model_config['z_dim'])
        ]
        self.discriminator = S_DiscreteZ_Discriminator(
            self.model_config['o_dim'],
            self.model_config['z_dim'],
            self.model_config['disc_hidden_layers']
        ).to(self.device)

        self.optimizers_policy  = [optim.Adam(self.policies[k].parameters(), self.lr) for k in range(self.z_dim)]
        self.optimizer_value    = optim.Adam(self.value.parameters(), self.lr)
        self.optimizer_disc     = optim.Adam(self.discriminator.parameters(), self.lr)
        self.workers            = self._init_workers(config['env_config'], config['seed'])

        self.p_z                = np.full(self.z_dim, 1/self.z_dim)
        self.current_trained_primitive      = 0
        self.total_step, self.total_episode = 0, 0

    def _init_workers(self, env_config: str, initial_seed: int, seed_increment: bool = True) -> List[RayWorker]:
        workers = []
        for i in range(self.num_workers):
            workers.append(
                RayWorker.remote(
                    worker_id = i, 
                    model_config = self.model_config,
                    env_config = env_config,
                    seed = initial_seed + i * seed_increment, 
                    rollout_episode = self.num_worker_rollout,
                )
            )
        return workers

    def _compute_discrimination_reward(self, obs: np.array, z: np.array) -> List[float]:
        obs_tensor  = torch.from_numpy(obs).float().to(self.device)
        z_tensor    = torch.from_numpy(z).type(torch.int64).to(self.device)
        p_z         = np.tile(self.p_z, len(z_tensor)).reshape(len(z_tensor), self.z_dim)
        p_z         = torch.from_numpy(p_z).to(self.device).float()

        logits      = self.discriminator(obs_tensor)
        probs       = F.softmax(logits, -1)
        P_z_s       = probs.gather(-1, z_tensor)
        log_P_z_s   = torch.log(P_z_s + 1e-6)
        log_P_z     = torch.log(p_z.gather(-1, z_tensor) + 1e-6)

        rewards     = (log_P_z_s.detach() - log_P_z.detach()).squeeze(-1).tolist()
        return rewards

    def rollout_update(self) -> Dict:
        for k in range(self.z_dim):
            policy = self.policies[k]
            
            # rollout and collect trajectories
            policy_state_dict_remote    = ray.put(deepcopy(policy).to(torch.device('cpu')).state_dict())
            rollout_remote              = [worker.rollout.remote(policy_state_dict_remote) for worker in self.workers]
            results                     = ray.get(rollout_remote)

            # unpack the data
            all_info_dict, all_trajectory, worker_rewards = [], [], []
            for item in results:
                all_info_dict.append(item[0])
                all_trajectory += item[1]                           # link the trajectories
        
            # unpack the train info
            for info_dict in all_info_dict:
                self.total_step += info_dict['step'] 
                worker_rewards.append(info_dict['rewards'])        
            self.total_episode += self.num_worker_rollout * self.num_workers
        
            # compute the hybrid rewards, return and advantages
            log_reward_in = 0
            for traj_dict in all_trajectory:
                if traj_dict['return'] > (1 - self.epsilon) * self.return_default:
                    # compute the discrimination reward
                    r_in_seq = self._compute_discrimination_reward(
                        np.stack(traj_dict['obs'], 0),
                        np.ones((len(traj_dict['obs']), 1)) * k
                    )
                    log_reward_in += np.mean(r_in_seq)
                    # compute the hybrid rewards
                    hybrid_r = [r_ex + self.alpha * r_in for r_ex, r_in in zip(traj_dict['r'], r_in_seq)]
                else:
                    hybrid_r = traj_dict['r']      

                value_seq = self.value(
                    torch.from_numpy(np.stack(traj_dict['obs'], 0)).to(self.device).float()
                ).squeeze(-1).detach().tolist()
                # compute the return and advantage 
                ret_seq, adv_seq = gae_estimator(hybrid_r, value_seq, self.gamma, self.lamda)               
                traj_dict.update({'ret': ret_seq, 'adv': adv_seq})
            log_reward_in /= len(all_trajectory)

            # unpack the trajectories
            data_buffer = {'obs': [], 'action': [], 'ret': [], 'adv': [], 'logprob': []}
            for traj_dict in all_trajectory:
                data_buffer['obs']              += traj_dict['obs']                         # [num, o_dim]
                data_buffer['action']           += traj_dict['action']                      # [num, a_dim]
                data_buffer['logprob']          += traj_dict['logprob']                     # [num, a_dim]
                data_buffer['ret']              += traj_dict['ret']                         # [num]
                data_buffer['adv']              += traj_dict['adv']                         # [num]

            for key in list(data_buffer.keys()):
                type_data = np.array(data_buffer[key], dtype=np.float64)
                if key in ['ret', 'adv']:
                    type_data = type_data.reshape(-1, 1)                                    # convert [num] to [num, 1]
                data_buffer[key] = torch.from_numpy(type_data).to(self.device).float()
            all_batch = Dataset(data_buffer)

            log_loss_value, log_loss_policy, log_loss_disc = 0, 0, 0
            update_count = 0
            for i in range(self.num_epoch):
                for batch in all_batch.iterate_once(self.batch_size):
                    obs, a, logprob, ret, adv = batch['obs'], batch['action'], batch['logprob'], batch['ret'], batch['adv']
                    
                    loss_value = self.compute_value_loss(obs, ret)
                    self.optimizer_value.zero_grad()
                    loss_value.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                    self.optimizer_value.step()

                    loss_policy = self.compute_policy_loss(k, obs, a, logprob, adv)
                    self.optimizers_policy[k].zero_grad()
                    loss_policy.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    self.optimizers_policy[k].step()

                    loss_disc = self.compute_discriminator_loss(obs, torch.ones((len(ret))).to(self.device) * k)
                    self.optimizer_disc.zero_grad()
                    loss_disc.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                    self.optimizer_disc.step()

                    update_count            += 1
                    log_loss_value          += loss_value.detach().item()
                    log_loss_policy         += loss_policy.detach().item()
                    log_loss_disc           += loss_disc.detach().item()
            
        return {
            'loss_policy':   log_loss_policy / update_count,
            'loss_disc':     log_loss_disc / update_count,
            'loss_value':    log_loss_value / update_count,
            'work_score':    float(np.mean(worker_rewards)),
            'reward_in':     log_reward_in
        }

    def compute_policy_loss(
        self, 
        k: int,
        obs_z: torch.tensor, 
        a: torch.tensor,
        logprob: torch.tensor, 
        adv: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        if len(adv) != 1:                                          # length is 1, the std will be nan
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        # compute the policy loss
        dist = self.policies[k](obs_z)
        new_logprob = dist.log_prob(a)
        entropy = dist.entropy()
        ratio = torch.exp(new_logprob.sum(-1, keepdim=True) - logprob.sum(-1, keepdim=True))
        surr1 = ratio * adv
        surr2 = torch.clip(ratio, 1-self.ratio_clip, 1+self.ratio_clip) * adv
        loss_policy = (- torch.min(surr1, surr2) - self.temperature_coef * entropy).mean()
        
        return loss_policy

    def compute_value_loss(self, obs_z: torch.tensor, ret: torch.tensor) -> torch.tensor:
        value = self.value(obs_z)
        loss_value = F.mse_loss(value, ret)
        return loss_value

    def compute_discriminator_loss(self, obs: torch.tensor, z: torch.tensor) -> torch.tensor:
        logits = self.discriminator(obs)
        loss_disc = F.cross_entropy(logits, z.type(torch.int64))
        return loss_disc
        
    def evaluate(self, env, num_episodes: int) -> float:
        rewards_across_skill = []
        for k in range(self.z_dim):
            reward = 0
            for i_episode in range(num_episodes):
                done = False
                obs = env.reset()
                step = 0
                while not done:
                    action = self.policies[k].act(torch.from_numpy(obs).float().to(self.device), False).detach().cpu().numpy()
                    action = np.clip(action, -env.action_bound, env.action_bound)
                    next_obs, r, done, info = env.step(action)
                    reward += r
                    obs = next_obs
                    step += 1
            rewards_across_skill.append(reward / num_episodes)
        return rewards_across_skill

    def save_policy(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        for k in range(self.z_dim):
            path = model_path + f'policy_{k}_{remark}'
            torch.save(self.policies[k].state_dict(), path)
        print(f"------- Policy saved to {model_path} as {remark} ----------")
    
    def save_discriminator(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'disc_{remark}'
        torch.save(self.discriminator.state_dict(), model_path)
        print(f"------- Discriminator saved to {model_path} ----------")

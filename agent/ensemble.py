from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
from state_filtration_for_qd.model.value import VFunction
from state_filtration_for_qd.model.primitive import Primitive
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.utils import gae_estimator, confirm_path_exist
from state_filtration_for_qd.buffer import Dataset


@ray.remote
class RayWorker(object):
    def __init__(
        self,
        worker_id: int,
        model_config: Dict,
        env_config: Dict,
        reward_tradeoff: float,
        seed: int,
        rollout_episode: int
    ) -> None:
        self.id = worker_id
        self.tradeoff = reward_tradeoff
        self.rollout_episode = rollout_episode

        self.policy = FixStdGaussianPolicy(
            model_config['o_dim'],
            model_config['a_dim'],
            model_config['policy_hidden_layers'],
            model_config['action_std']
        )
        self.env = call_env(env_config)
        self.env.seed(seed)
        self.obs_filter = self.env._process_obs
        self.action_bound = self.env.action_bound

    def rollout(self, policy_state_dict: nn.ModuleDict) -> Tuple:
        self.policy.load_state_dict(policy_state_dict)

        info_dict = {'step': 0, 'rewards': 0}
        trajectories = [dict() for _ in range(self.rollout_episode)]
        for i_episode in range(self.rollout_episode):
            obs_seq, next_obs_seq, a_seq, r_seq, logprob_seq = [], [], [], [], []
            filtrated_obs_seq, filtrated_next_obs_seq = [], []

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
                obs_filt, next_obs_filt = self.obs_filter(obs), self.obs_filter(obs_)
                
                # pack the trajectory
                obs_seq.append(obs)
                next_obs_seq.append(obs_)
                a_seq.append(action)
                r_seq.append(r)
                logprob_seq.append(logprob)
                filtrated_obs_seq.append(obs_filt)
                filtrated_next_obs_seq.append(next_obs_filt)

                info_dict['step'] += 1
                info_dict['rewards'] += r
                obs = obs_

            trajectories[i_episode].update({
                'obs': obs_seq,
                'action': a_seq,
                'next_obs': next_obs_seq,
                'r': r_seq,
                'logprob': logprob_seq,
                'obs_filt': filtrated_obs_seq,
                'next_obs_filt': filtrated_next_obs_seq
            })
            info_dict['rewards'] /= self.rollout_episode
        
        return (info_dict, trajectories)


class Ensemble(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.model_config = config['model_config']
        self.tradeoff = config['reward_tradeoff']
        self.num_primitive = config['num_primitive']
        self.exp_path = config['exp_path']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.lamda = config['lamda']
        self.ratio_clip = config['ratio_clip']
        self.temperature_coef = config['temperature_coef']
        self.num_epoch = config['num_epoch']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.num_worker_rollout = config['num_worker_rollout']
        self.device = torch.device(config['device'])

        self.value = VFunction(
            self.model_config['o_dim'],
            self.model_config['value_hidden_layers']
        ).to(self.device)
        self.optimizer_value = optim.Adam(self.value.parameters(), self.lr)
        self.primitives, self.primitive_policy_optimizers, self.primitive_inverse_model_optimizers = self._init_primitives()
        
        self.workers = self._init_workers(config['env_config'], config['seed'])

        self.current_trained_primitive      = 0
        self.total_step, self.total_episode = 0, 0

    def _init_primitives(self) -> Tuple:
        primitives                          = []
        primitive_policy_optimizers         = []
        primitive_inverse_model_optimizers  = []
        for i in range(self.num_primitive):
            p                   = Primitive(self.model_config, self.device)
            opt_policy          = optim.Adam(p.policy.parameters(), self.lr)
            opt_inverse_model   = optim.Adam(p.inverse_model.parameters(), self.lr, weight_decay=0.01)

            primitives.append(p)
            primitive_policy_optimizers.append(opt_policy)
            primitive_inverse_model_optimizers.append(opt_inverse_model)

        return primitives, primitive_policy_optimizers, primitive_inverse_model_optimizers
    
    def _init_workers(self, env_config: str, initial_seed: int, seed_increment: bool = True) -> List[RayWorker]:
        workers = []
        for i in range(self.num_workers):
            workers.append(
                RayWorker.remote(
                    worker_id = i, 
                    model_config = self.model_config,
                    env_config = env_config,
                    reward_tradeoff = self.tradeoff,
                    seed = initial_seed + i * seed_increment, 
                    rollout_episode = self.num_worker_rollout,
                )
            )
        return workers
    
    def _compute_disagree_reward(self, obs_filt: np.array, next_obs_filt: np.array, action: np.array) -> List[float]:
        intrinsic_reward = np.zeros((len(obs_filt),), dtype=np.float64)
        if self.current_trained_primitive == 0:
            return intrinsic_reward.tolist()
        else:
            for k in range(self.current_trained_primitive):
                inference_a_dist = self.primitives[k].inference_action(obs_filt, next_obs_filt)
                log_likelihood = - inference_a_dist.log_prob(
                    torch.from_numpy(action).to(self.device).float()
                ).sum(-1, keepdim=False).detach().cpu().numpy()
                intrinsic_reward += log_likelihood
            intrinsic_reward /= self.current_trained_primitive
            return intrinsic_reward.tolist()

    def rollout_update(self, current_primitive: int) -> Dict:
        self.current_trained_primitive          = current_primitive
        primitive                               = self.primitives[current_primitive]
        primitive_policy_optimizer              = self.primitive_policy_optimizers[current_primitive]
        primitive_inverse_model_optimizer       = self.primitive_inverse_model_optimizers[current_primitive]
       
        # rollout and collect trajectories
        policy_state_dict_remote    = ray.put(deepcopy(primitive.policy).to(torch.device('cpu')).state_dict())
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
            # compute the disagree reward
            r_in_seq = self._compute_disagree_reward(
                np.stack(traj_dict['obs_filt'], 0),
                np.stack(traj_dict['next_obs_filt'], 0),
                np.stack(traj_dict['action'], 0)
            )
            log_reward_in += np.mean(r_in_seq)
            # compute the hybrid rewards
            hybrid_r = [r_ex + self.tradeoff * r_in for r_ex, r_in in zip(traj_dict['r'], r_in_seq)]
            value_seq = self.value(
                torch.from_numpy(np.stack(traj_dict['obs'], 0)).to(self.device).float()
            ).squeeze(-1).detach().tolist()
            # compute the return and advantage 
            ret_seq, adv_seq = gae_estimator(hybrid_r, value_seq, self.gamma, self.lamda)
            traj_dict.update({'ret': ret_seq, 'adv': adv_seq})
        log_reward_in /= len(all_trajectory)

        # unpack the trajectories
        data_buffer = {'obs': [], 'action': [], 'ret': [], 'adv': [], 'logprob': [], 'obs_filt': [], 'next_obs_filt': []}
        for traj_dict in all_trajectory:
            data_buffer['obs']              += traj_dict['obs']                         # [num, o_dim]
            data_buffer['action']           += traj_dict['action']                      # [num, a_dim]
            data_buffer['logprob']          += traj_dict['logprob']                     # [num, a_dim]
            data_buffer['obs_filt']         += traj_dict['obs_filt']                    # [num, o_dim]
            data_buffer['next_obs_filt']    += traj_dict['next_obs_filt']               # [num, o_dim]
            data_buffer['ret']              += traj_dict['ret']                         # [num]
            data_buffer['adv']              += traj_dict['adv']                         # [num]
        for key in list(data_buffer.keys()):
            type_data = np.array(data_buffer[key], dtype=np.float64)
            if key in ['ret', 'adv']:
                type_data = type_data.reshape(-1, 1)                    # convert [num] to [num, 1]
            data_buffer[key] = torch.from_numpy(type_data).to(self.device).float()
        all_batch = Dataset(data_buffer)

        log_loss_value, log_loss_policy, log_loss_inverse_model = 0, 0, 0
        update_count = 0
        for i in range(self.num_epoch):
            for batch in all_batch.iterate_once(self.batch_size):
                obs, a, logprob, ret, adv = batch['obs'], batch['action'], batch['logprob'], batch['ret'], batch['adv']
                obs_filt, next_obs_filt = batch['obs_filt'], batch['next_obs_filt']
                
                loss_value = self.compute_value_loss(obs, ret)
                self.optimizer_value.zero_grad()
                loss_value.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer_value.step()

                loss_policy = self.compute_policy_loss(primitive, obs, a, logprob, adv)
                primitive_policy_optimizer.zero_grad()
                loss_policy.backward()
                nn.utils.clip_grad_norm_(primitive.policy.parameters(), 0.5)
                primitive_policy_optimizer.step()

                loss_inverse_model = self.compute_inverse_model_loss(primitive, a, obs_filt, next_obs_filt)
                primitive_inverse_model_optimizer.zero_grad()
                loss_inverse_model.backward()
                nn.utils.clip_grad_norm_(primitive.inverse_model.parameters(), 0.5)
                primitive_inverse_model_optimizer.step()

                update_count            += 1
                log_loss_value          += loss_value.detach().item()
                log_loss_policy         += loss_policy.detach().item()
                log_loss_inverse_model  += loss_inverse_model#.detach().item()
        
        return {
            'loss_policy': log_loss_policy / update_count,
            'loss_inverse_model': log_loss_inverse_model / update_count,
            'loss_value': log_loss_value / update_count,
            'work_score': float(np.mean(worker_rewards)),
            'reward_in': log_reward_in
        }

    def compute_policy_loss(
        self, 
        primitive: Primitive, 
        obs: torch.tensor, 
        a: torch.tensor, 
        logprob: torch.tensor, 
        adv: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        if len(adv) != 1:   # length is 1, the std will be nan
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        # compute the policy loss
        dist = primitive.policy(obs)
        new_logprob = dist.log_prob(a)
        entropy = dist.entropy()
        ratio = torch.exp(new_logprob.sum(-1, keepdim=True) - logprob.sum(-1, keepdim=True))
        surr1 = ratio * adv
        surr2 = torch.clip(ratio, 1-self.ratio_clip, 1+self.ratio_clip) * adv
        loss_policy = (- torch.min(surr1, surr2) - self.temperature_coef * entropy).mean()
        
        return loss_policy
    
    def compute_inverse_model_loss(
        self, 
        primitive: Primitive, 
        a: torch.tensor,
        obs_filt: torch.tensor,
        next_obs_filt: torch.tensor
    ) -> torch.tensor:
        inference_a_dist = primitive.inverse_model(obs_filt, next_obs_filt)
        loss_inverse_model = - inference_a_dist.log_prob(a).mean()
        return loss_inverse_model

    def compute_value_loss(self, obs: torch.tensor, ret: torch.tensor) -> torch.tensor:
        value = self.value(obs)
        loss_value = F.mse_loss(value, ret)
        return loss_value
    
    def evaluate(self, env, num_episodes: int) -> float:
        # evaluate the current trained primitive
        k = self.current_trained_primitive
        reward = 0
        for i_episode in range(num_episodes):
            done = False
            obs = env.reset()
            step = 0
            while not done:
                action = self.primitives[k].decision(obs, False)
                action = np.clip(action, -env.action_bound, env.action_bound)
                next_obs, r, done, info = env.step(action)
                reward += r
                obs = next_obs
                step += 1
        return reward / num_episodes

    def save_policy(self, id_primitive: int, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'policy_{id_primitive}_{remark}'
        torch.save(self.primitives[id_primitive].policy.state_dict(), model_path)
        print(f"- - - - Policy of primitive {id_primitive} saved to {model_path} - - - -")
    
    def save_inverse_model(self, id_primitive: int, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'inverse_model_{id_primitive}_{remark}'
        torch.save(self.primitives[id_primitive].inverse_model.state_dict(), model_path)
        print(f"- - - - Inverse model of primitive {id_primitive} saved to {model_path} - - - -")

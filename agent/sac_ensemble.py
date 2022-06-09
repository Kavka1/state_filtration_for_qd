from turtle import back
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from state_filtration_for_qd.utils import soft_update, hard_update, confirm_path_exist
from state_filtration_for_qd.buffer import Buffer
from state_filtration_for_qd.model.latent_policy import Latent_DiagGaussianPolicy
from state_filtration_for_qd.model.dynamics import LatentDiagGaussianIDM
from state_filtration_for_qd.model.value import TwinQFunction


class Sac_Ensemble(object):
    def __init__(self, config: Dict) -> None:
        super(Sac_Ensemble, self).__init__()
        self.model_config       = config['model_config']
        self.z_dim              = self.model_config['z_dim']
        self.lr                 = config['lr']
        self.gamma              = config['gamma']
        self.tau                = config['tau']
        self.batch_size         = config['batch_size']
        self.train_policy_delay = config['train_policy_delay']
        self.tradeoff_ex        = config['reward_tradeoff_ex']
        self.tradeoff_in        = config['reward_tradeoff_in']
        self.reward_in_start    = config['reward_in_start_step']
        self.exp_path           = config['exp_path']
        self.device             = torch.device(config['device'])

        self.target_entropy             = - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
        self.p_z                        = np.tile(np.full(self.z_dim, 1/self.z_dim), self.batch_size).reshape(self.batch_size, self.z_dim)
        # prepare the matrix for compute disagree of IDMs [10, 9, 10]
        prepare_one_hot_z_expand   = []
        for z in range(self.z_dim):
            temp_matrix = np.zeros((self.z_dim, self.z_dim))
            for z_small in range(self.z_dim):
                temp_matrix[z_small,:]  =  (np.arange(self.z_dim) == z_small).astype(np.integer)
            temp_matrix = np.delete(temp_matrix, z, 0)
            prepare_one_hot_z_expand.append(temp_matrix)
        self.prepare_one_hot_z_expand = np.stack(prepare_one_hot_z_expand, 0)

        if 'fix_alpha' in list(config.keys()):
            self.train_alpha = False
            self.alpha = torch.tensor(config['fix_alpha']).type(torch.float64)
            self.log_alpha = torch.log(self.alpha)
        else:
            self.train_alpha = True
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = torch.exp(self.log_alpha)

        self._init_model()
        self._init_optimizer()
        self._init_logger()

    def _init_model(self) -> None:
        o_dim, a_dim, z_dim     = self.model_config['o_dim'], self.model_config['a_dim'], self.model_config['z_dim']
        policy_hiddens          = self.model_config['policy_hidden_layers']
        value_hiddens           = self.model_config['value_hidden_layers']
        idm_hiddens             = self.model_config['idm_hidden_layers']
        idm_input_delta         = self.model_config['idm_input_delta']

        policy_logstd_min       = self.model_config['policy_logstd_min']
        policy_logstd_max       = self.model_config['policy_logstd_max']
        idm_logstd_min          = self.model_config['idm_logstd_min']
        idm_logstd_max          = self.model_config['idm_logstd_max']

        self.policy = Latent_DiagGaussianPolicy(o_dim, a_dim, z_dim, policy_hiddens, policy_logstd_min, policy_logstd_max).to(self.device)
        self.inverse_dynamics = LatentDiagGaussianIDM(o_dim, a_dim, z_dim, idm_input_delta, idm_hiddens, idm_logstd_min, idm_logstd_max).to(self.device)
        self.value = TwinQFunction(o_dim + z_dim, a_dim, value_hiddens).to(self.device)
        self.value_tar = TwinQFunction(o_dim + z_dim, a_dim, value_hiddens).to(self.device)
        hard_update(self.value, self.value_tar)

    def _init_optimizer(self) -> None:
        self.optimizer_policy   = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_idm      = optim.Adam(self.inverse_dynamics.parameters(), lr=self.lr)
        self.optimizer_value    = optim.Adam(self.value.parameters(), lr=self.lr)
        self.optimizer_alpha    = optim.Adam([self.log_alpha], lr=self.lr)

    def _init_logger(self) -> None:
        self.logger_loss_value  = 0.
        self.logger_loss_policy = 0.
        self.logger_loss_idm    = 0.
        self.logger_loss_alpha  = 0.
        self.logger_alpha       = self.alpha.item()
        self.update_count       = 0

    def choose_action(self, obs_z: np.array, with_noise: bool) -> np.array:
        obs_z = torch.from_numpy(obs_z).float().to(self.device)
        action = self.policy.act(obs_z, with_noise)
        return action.detach().cpu().numpy()

    def compute_value_loss(self, obs_z: torch.Tensor, a: torch.Tensor, r: torch.Tensor, done: torch.Tensor, next_obs_z: torch.Tensor) -> torch.tensor:
        with torch.no_grad():
            next_a, next_a_logprob, dist = self.policy(obs_z)
            next_q1_tar, next_q2_tar = self.value_tar(next_obs_z, next_a)
            next_q_tar = torch.min(next_q1_tar, next_q2_tar)
            q_update_tar = r + (1 - done) * self.gamma * (next_q_tar - self.alpha * next_a_logprob)
        q1_pred, q2_pred = self.value(obs_z, a)
        loss_q = F.mse_loss(q1_pred, q_update_tar) + F.mse_loss(q2_pred, q_update_tar)
        return loss_q

    def compute_policy_alpha_loss(self, obs_z: torch.Tensor) -> Tuple:
        a_new, a_new_logprob, dist_new = self.policy(obs_z)
        loss_policy = (self.alpha * a_new_logprob - self.value.call_Q1(obs_z, a_new)).mean()
        # Dual policy gradient
        a_new_logprob = torch.tensor(a_new_logprob.tolist(), requires_grad=False, device=self.device)
        loss_alpha = (- torch.exp(self.log_alpha) * (a_new_logprob + self.target_entropy)).mean()
        return loss_policy, loss_alpha

    def compute_idm_loss(self, obs_filt: torch.tensor, next_obs_filt: torch.tensor, a: torch.tensor, z_one_hot: torch.tensor) -> torch.tensor:
        dist        = self.inverse_dynamics(obs_filt, next_obs_filt, z_one_hot)
        loss_idm    = - dist.log_prob(a).mean()
        return loss_idm

    def unpack(self, obs: np.array, obs_filt: np.array, a: np.array, r: np.array, done: np.array, obs_: np.array, next_obs_filt: np.array, z: np.array,) -> Tuple:
        # build one-hot z
        z_one_hot   = (np.arange(self.model_config['z_dim']) == z[:, None]).astype(np.integer)
        # array to tensor
        device          = self.device
        obs             = torch.from_numpy(obs).to(device).float()
        obs_filt        = torch.from_numpy(obs_filt).to(device).float()
        a               = torch.from_numpy(a).to(device).float()
        r               = torch.from_numpy(r).to(device).float().unsqueeze(-1)
        done            = torch.from_numpy(done).to(device).float().unsqueeze(-1)
        obs_            = torch.from_numpy(obs_).to(device).float()
        next_obs_filt   = torch.from_numpy(next_obs_filt).to(device).float()
        z               = torch.from_numpy(z).to(device).type(torch.int64).unsqueeze(-1)
        z_one_hot       = torch.from_numpy(z_one_hot).to(device).float()

        return obs, obs_filt, a, r, done, obs_, next_obs_filt, z, z_one_hot

    def train(self, buffer: Buffer) -> Dict:
        if len(buffer) < self.batch_size:
            return {
                'loss_value': 0, 
                'loss_policy': 0, 
                'loss_alpha': 0, 
                'loss_idm': 0,
                'avg_bonus': 0,
                'alpha': self.logger_alpha
            }
        obs, obs_filt, a, r, done, obs_, next_obs_filt, z               = buffer.sample(self.batch_size)
        obs, obs_filt, a, r, done, obs_, next_obs_filt, z, z_one_hot    = self.unpack(obs, obs_filt, a, r, done, obs_, next_obs_filt, z)
        obs_z           = torch.cat([obs, z_one_hot], -1)
        next_obs_z      = torch.cat([obs_, z_one_hot], -1)
        
        #   compute intrinsic rewards
        if self.update_count > self.reward_in_start:
            with torch.no_grad():
                expanded_obs_filt       = torch.reshape(
                    obs_filt.repeat(1, self.z_dim - 1), 
                    (self.batch_size, self.z_dim-1, self.model_config['o_dim']))
                expanded_next_obs_filt  = torch.reshape(
                    next_obs_filt.repeat(1, self.z_dim - 1),
                    (self.batch_size, self.z_dim-1, self.model_config['o_dim']))
                expanded_a              = torch.reshape(
                    a.repeat(1, self.z_dim-1),
                    (self.batch_size, self.z_dim-1, self.model_config['a_dim']))
                
                temp_z_one_hot_batch    = []
                for single_z in z:
                    z_idx               = single_z.detach().tolist()[0]
                    temp_z_one_hot_batch.append(np.copy(self.prepare_one_hot_z_expand[z_idx]))
                temp_z_one_hot_batch    = np.stack(temp_z_one_hot_batch, 0)
                expanded_z_one_hot      = torch.from_numpy(temp_z_one_hot_batch).to(self.device).float()
                
                inference_dist  = self.inverse_dynamics(expanded_obs_filt, expanded_next_obs_filt, expanded_z_one_hot)  # [B, |Z|-1, |Z|]
                log_likelihood  = inference_dist.log_prob(expanded_a).mean(-1, keepdim=False)   # [B, |Z|-1]
                bonus           = torch.mean(-log_likelihood, -1, keepdim=True)
        else:
            bonus = torch.zeros_like(r).to(self.device).float()
        r   =   self.tradeoff_ex * r + self.tradeoff_in * bonus

        #   train value function
        loss_value = self.compute_value_loss(obs_z, a, r, done, next_obs_z)
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()

        #   train discriminator
        loss_idm  = self.compute_idm_loss(obs_filt, next_obs_filt, a, z_one_hot)
        self.optimizer_idm.zero_grad()
        loss_idm.backward()
        nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), 0.5)
        self.optimizer_idm.step()

        if self.update_count % self.train_policy_delay == 0:
            loss_policy, loss_alpha = self.compute_policy_alpha_loss(obs_z)
            # train policy
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer_policy.step()

            if self.train_alpha:
                self.optimizer_alpha.zero_grad()
                loss_alpha.backward()
                self.optimizer_alpha.step()
                self.alpha = torch.exp(self.log_alpha)

                self.logger_alpha = self.alpha.detach().item()   
                self.logger_loss_alpha = loss_alpha.detach().item()
            else:
                self.logger_alpha = self.alpha.item()   
                self.logger_loss_alpha = loss_alpha.item()            

            self.logger_loss_policy = loss_policy.detach().item()

        soft_update(self.value, self.value_tar, self.tau)

        self.logger_loss_value  = loss_value.detach().item()
        self.logger_loss_idm    = loss_idm.detach().item()
        self.update_count += 1
        return {
            'loss_value':       self.logger_loss_value, 
            'loss_idm'  :       self.logger_loss_idm,
            'loss_policy':      self.logger_loss_policy, 
            'loss_alpha':       self.logger_loss_alpha, 
            'avg_bonus':        torch.mean(bonus).detach().item(),
            'alpha':            self.logger_alpha
        }

    def evaluate(self, env, num_episodes: int) -> float:
        rewards_across_skill = []
        for z in range(self.z_dim):
            reward = 0
            one_hot_z = np.zeros(self.z_dim).astype(np.float64)
            one_hot_z[z] = 1
            for i_episode in range(num_episodes):
                done = False
                obs = env.reset()
                step = 0
                while not done:
                    action = self.choose_action(np.concatenate([obs, one_hot_z],-1), False)
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
    
    def save_idm(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'disc_{remark}'
        torch.save(self.inverse_dynamics.state_dict(), model_path)
        print(f"------- IDM saved to {model_path} ----------")

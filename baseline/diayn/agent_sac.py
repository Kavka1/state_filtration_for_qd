from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from state_filtration_for_qd.utils import soft_update, hard_update, confirm_path_exist
from state_filtration_for_qd.buffer import Buffer
from state_filtration_for_qd.model.latent_policy import Latent_DiagGaussianPolicy
from state_filtration_for_qd.model.discriminator import S_DiscreteZ_Discriminator
from state_filtration_for_qd.model.value import TwinQFunction


class DIAYN_SAC(object):
    def __init__(self, config: Dict) -> None:
        super(DIAYN_SAC, self).__init__()
        self.model_config       = config['model_config']
        self.z_dim              =   self.model_config['z_dim']
        self.lr                 = config['lr']
        self.gamma              = config['gamma']
        self.tau                = config['tau']
        self.batch_size         = config['batch_size']
        self.train_policy_delay = config['train_policy_delay']
        self.device             = torch.device(config['device'])
        self.tradeoff_ex        = config['reward_tradeoff_ex']
        self.tradeoff_in        = config['reward_tradeoff_in']

        self.target_entropy = - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
        self.p_z            = np.tile(np.full(self.z_dim, 1/self.z_dim), self.batch_size).reshape(self.batch_size, self.z_dim)
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
        disc_hiddens            = self.model_config['disc_hidden_layers']
        logstd_min, logstd_max  = self.model_config['policy_logstd_min'], self.model_config['policy_logstd_max']

        self.policy = Latent_DiagGaussianPolicy(
            o_dim, a_dim, z_dim, policy_hiddens, logstd_min, logstd_max,
        ).to(self.device)
        self.discriminator = S_DiscreteZ_Discriminator(
            o_dim, z_dim, disc_hiddens
        ).to(self.device)
        self.value = TwinQFunction(
            o_dim + z_dim, a_dim, value_hiddens
        ).to(self.device)
        self.value_tar = TwinQFunction(
            o_dim + z_dim, a_dim, value_hiddens
        ).to(self.device)
        hard_update(self.value, self.value_tar)

    def _init_optimizer(self) -> None:
        self.optimizer_policy   = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_disc     = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.optimizer_value    = optim.Adam(self.value.parameters(), lr=self.lr)
        self.optimizer_alpha    = optim.Adam([self.log_alpha], lr=self.lr)

    def _init_logger(self) -> None:
        self.logger_loss_value  = 0.
        self.logger_loss_policy = 0.
        self.logger_loss_disc   = 0.
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

    def compute_disc_loss(self, obs: torch.tensor, z: torch.tensor) -> torch.tensor:
        logits      = self.discriminator(obs)
        probs       = F.softmax(logits, -1)
        log_probs   = torch.log(probs.gather(-1, z))
        loss_disc   = F.cross_entropy(log_probs, z.squeeze(-1))
        return loss_disc

    def unpack(self, obs: np.array, a: np.array, r: np.array, done: np.array, obs_: np.array, z: np.array,) -> Tuple:
        # build one-hot z
        z_one_hot   = (np.arange(self.model_config['z_dim']) == z[:, None]).astype(np.integer)
        # array to tensor
        device      = self.device
        obs         = torch.from_numpy(obs).to(device).float()
        a           = torch.from_numpy(a).to(device).float()
        r           = torch.from_numpy(r).to(device).float().unsqueeze(-1)
        done        = torch.from_numpy(done).to(device).int().unsqueeze(-1)
        obs_        = torch.from_numpy(obs_).to(device).float()
        z           = torch.from_numpy(z).to(device).type(torch.int64).unsqueeze(-1)
        z_one_hot   = torch.from_numpy(z_one_hot).to(device).float()

        return obs, a, r, done, obs_, z, z_one_hot

    def train(self, buffer: Buffer) -> Dict:
        if len(buffer) < self.batch_size:
            return {
                'loss_value': 0, 
                'loss_policy': 0, 
                'loss_alpha': 0, 
                'loss_disc': 0,
                'avg_bonus': 0,
                'alpha': self.logger_alpha
            }
        obs, a, r, done, obs_, z                = buffer.sample(self.batch_size)
        obs, a, r, done, obs_, z, z_one_hot     = self.unpack(obs, a, r, done, obs_, z)
        obs_z           = torch.cat([obs, z_one_hot], -1)
        next_obs_z      = torch.cat([obs_, z_one_hot])
        
        #   compute intrinsic rewards
        with torch.no_grad():
            p_z                 = torch.from_numpy(self.p_z).to(self.device)
            logits              =   self.discriminator(obs)
            probs               =   F.softmax(logits, -1)
            log_P_z_given_s     =   torch.log(probs.gather(-1, z))
            log_P_z             =   torch.log(p_z.gather(-1, z) + 1e-6)
            bonus               =   log_P_z_given_s - log_P_z
        r   =   self.tradeoff_ex * r + self.tradeoff_in * bonus

        #   train value function
        loss_value = self.compute_value_loss(obs_z, a, r, done, next_obs_z)
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()

        #   train discriminator
        loss_disc  = self.compute_disc_loss(obs, z)
        self.optimizer_disc.zero_grad()
        loss_disc.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.optimizer_disc.step()

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
        self.logger_loss_disc   = loss_disc.detach().item()
        self.update_count += 1
        return {
            'loss_q':       self.logger_loss_q, 
            'loss_disc':    self.logger_loss_disc,
            'loss_policy':  self.logger_loss_policy, 
            'loss_alpha':   self.logger_loss_alpha, 
            'avg_bonus':    torch.mean(bonus).detach().item(),
            'alpha':        self.logger_alpha
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
    
    def save_discriminator(self, remark: str) -> None:
        model_path = self.exp_path + 'model/'
        confirm_path_exist(model_path)
        model_path = model_path + f'disc_{remark}'
        torch.save(self.discriminator.state_dict(), model_path)
        print(f"------- Discriminator saved to {model_path} ----------")

from operator import index
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
import gym
import yaml
from torch.utils.tensorboard import SummaryWriter

from state_filtration_for_qd.utils import confirm_path_exist, make_exp_path
from state_filtration_for_qd.agent.ppo import PPOAgent
from state_filtration_for_qd.model.latent_policy import Latent_FixStdGaussianPolicy
from state_filtration_for_qd.logger import Logger
from state_filtration_for_qd.env.common import call_env


def main(config: Dict, exp_name: str = ''):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # make experiment file dir
    make_exp_path(config, exp_name)
    # build env
    env = call_env(config['env_config'])
    # update config about env and save config
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0],
        'filtrated_o_dim': len(env._process_obs(env.reset()))
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)
    # build agent
    agent = PPOAgent(config)
    agent._init_workers(
        env_config= config['env_config'],
        initial_seed= config['seed'],
        rollout_episodes= config['rollout_episode'],
        seed_increment= True
    )
    # make loggers
    logger = Logger()
    tb = SummaryWriter(config['exp_path'])

    total_step, total_episode, total_iteration = 0, 0, 0
    best_score, best_disc = 0, 1
    while total_step <= config['max_timesteps']:
        logger_dict = agent.roll_update()

        total_step = agent.total_steps
        total_episode = agent.total_episodes

        rewards_across_skill = agent.evaluate(env, config['eval_episode'])
        max_skill_score = np.max(rewards_across_skill)
        mean_skill_score = np.mean(rewards_across_skill)
        min_skill_score = np.min(rewards_across_skill)

        disc_accuracy = logger_dict['loss_disc']
        if max_skill_score > best_score:
            agent.save_policy('best')
            best_score = max_skill_score
        if disc_accuracy < best_disc:
            agent.save_discriminator('best')
            best_disc = disc_accuracy
            
        logger_dict.update({
            'episode': total_episode, 
            'steps': total_step, 
            'eval_return': max_skill_score,
            'max_skill_score': max_skill_score,
            'mean_skill_score': mean_skill_score,
            'min_skill_score': min_skill_score
        })
        logger.store(**logger_dict)
        logger.save(config['exp_path'] + 'log.pkl')
        # Log the results in terminal
        print("| Step {} | Episode {} | {}".format(total_step, total_episode, logger))
        # Log the results in tb
        tb.add_scalar('Eval/eval_return', max_skill_score, total_step)
        tb.add_scalar('Eval/max_skill_score', max_skill_score, total_step)
        tb.add_scalar('Eval/mean_skill_score', mean_skill_score, total_step)
        tb.add_scalar('Eval/min_skill_score', min_skill_score, total_step)
        for name, value_seq in list(logger.data.items()):
            tb.add_scalar(f'Train/{name}', value_seq[-1], total_step)
        # save models periodically
        if total_iteration % config['save_interval'] == 0:
            agent.save_policy(f'{total_iteration}_{total_step}')
            agent.save_discriminator(f'{total_iteration}_{total_step}')

        total_iteration += 1


def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    policy = Latent_FixStdGaussianPolicy(
        config['model_config']['o_dim'],
        config['model_config']['a_dim'],
        config['model_config']['z_dim'],
        config['model_config']['policy_hidden_layers'],
        config['model_config']['action_std'],
    )

    policy.load_model(path + f'model/policy_{remark}')
    env = call_env(config['env_config'])

    z_dim = config['model_config']['z_dim']
    for _ in range(100):
        for z in range(z_dim):
            z_one_hot = np.zeros(z_dim).astype(np.float64)
            z_one_hot[z] = 1.
            done = False
            episode_r = 0
            obs = env.reset()
            while not done:
                env.render()
                obs_z = np.concatenate([obs, z_one_hot], -1)
                obs_z = torch.from_numpy(obs_z).float()
                a = policy.act(obs_z, False).detach().numpy()
                obs, r, done, info = env.step(a)
                episode_r += r
            
            print(f"Skill {z} Episode {_} Episode Reward {episode_r}")



if __name__ == '__main__':
    config = {
        'seed': 10,

        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'z_dim': 20,
            'policy_hidden_layers': [256, 256],
            'value_hidden_layers': [256, 256],
            'disc_hidden_layers': [256, 256],
            'action_std': 0.4,
        },
        'env_config': {
            'env_name': 'Walker',
            'missing_obs_info': {
                'missing_joint': [],#['foot', 'leg', 'thigh'],
                'missing_coord': ['2', '3']
            }
        },

        'max_timesteps': 10000000,
        'save_interval': 100,
        'eval_episode': 1,
        
        'num_workers': 10,
        'rollout_episode': 5,
        'reward_tradeoff': 0.01,
        'num_epoch': 20,
        'lr': 0.0003,
        'gamma': 0.99,
        'lamda': 0.95,
        'ratio_clip': 0.25,
        'batch_size': 256,
        'temperature_coef': 0.1,
        'device': 'cpu',
        'result_path': '/home/xukang/Project/state_filtration_for_qd/results/'
    }
    
    for seed in [10, 20, 30]:
        config['seed'] = seed
        #main(config, 'tradeoff_0.01')

    demo('/home/xukang/Project/state_filtration_for_qd/results/tradeoff_0.01-Walker-missing_coord_2_3-10/','best')
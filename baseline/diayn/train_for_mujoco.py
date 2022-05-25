from typing import List, Dict, Tuple
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from state_filtration_for_qd.utils import confirm_path_exist, make_exp_path
from state_filtration_for_qd.logger import Logger
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.baseline.diayn.agent import DIAYN

from model.latent_policy import Latent_FixStdGaussianPolicy

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
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)
    # build agent
    agent = DIAYN(config)
    # make loggers
    logger = Logger()
    tb = SummaryWriter(config['exp_path'])

    total_step, total_episode, total_iteration = 0, 0, 0
    best_score = -100
    while total_step <= config['max_timesteps']:
        logger_dict = agent.rollout_update()

        total_step = agent.total_step
        total_episode = agent.total_episode

        reward_across_skills = agent.evaluate(env, config['eval_episode'])
        max_skill_score      = np.max(reward_across_skills)
        mean_skill_score     = np.mean(reward_across_skills)
        min_skill_score      = np.min(reward_across_skills)
        if max_skill_score > best_score:
            agent.save_policy('best')
            best_score = max_skill_score
        
        logger_dict.update({
            'iteration': total_iteration,
            'episode': total_episode, 
            'steps': total_step, 
            'max_skill_score': max_skill_score,
            'mean_skill_score': mean_skill_score,
            'min_skill_score': min_skill_score
        })
        
        logger.store(**logger_dict)   
        # Log the results in terminal
        print("| Iteration {} | Step {} | {}".format(total_iteration, total_step, logger))
        # Log the evaluation results in tb
        tb.add_scalar('Eval/max_skill_score', max_skill_score, total_iteration)
        tb.add_scalar('Eval/min_skill_score', min_skill_score, total_iteration)
        tb.add_scalar('Eval/mean_skill_score', mean_skill_score, total_iteration)
        tb.add_scalar('Eval/total_step', total_step, total_iteration)
        # Log the information of the training process in tb
        for name, value_seq in list(logger.data.items()):
            tb.add_scalar(f'Train/{name}', value_seq[-1], total_step)
        
        # save log
        if total_iteration % config['log_interval'] == 0:
            logger.save(config['exp_path'] + 'log.pkl')

        # save models periodically
        if total_iteration % config['save_interval'] == 0:
            agent.save_policy(f'{total_iteration}')
            agent.save_inverse_model(f'{total_iteration}')

        total_iteration += 1

    agent.save_policy('final')
    agent.save_inverse_model('final')


def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    z_dim =  config['model_config']['z_dim']

    policy = Latent_FixStdGaussianPolicy(
        config['model_config']['o_dim'],
        config['model_config']['a_dim'],
        config['model_config']['z_dim'],
        config['model_config']['policy_hidden_layers'],
        config['model_config']['action_std'],
    )
    policy.load_model(path + f'model/policy_{remark}')
    
    env = call_env(config['env_config'])
    
    for _ in range(100):
        for z in range(z_dim):
            reward = 0
            one_hot_z = np.zeros(z_dim).astype(np.float64)
            one_hot_z[z] = 1
            
            done = False
            obs = env.reset()
            step = 0
            while not done:
                obs_z_tensor = torch.from_numpy(np.concatenate([obs, one_hot_z], -1)).float()
                action = policy.act(obs_z_tensor, False).detach().numpy()
                next_obs, r, done, info = env.step(action)
                reward += r
                obs = next_obs
                step += 1
            
            print(f"Skill {z} Episode {_} Episode Reward {reward}")


if __name__ == '__main__':
    config = {
        'seed': 10,

        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'z_dim': 10,
            'policy_hidden_layers': [128, 128],
            'value_hidden_layers': [128, 128],
            'disc_hidden_layers': [128, 128],
            'action_std': 0.4,
        },
        'env_config': {
            'env_name': 'HalfCheetah',
            'missing_obs_info': {
                'missing_joint':    [],
                'missing_coord':    [],
                'missing_leg':      []
            }
        },

        'max_timesteps': 10000000,
        'save_interval': 40,
        'log_interval': 10,
        'eval_episode': 1,
        
        'num_workers': 10,
        'num_worker_rollout': 5,
        'reward_tradeoff': 0.05,
        'num_epoch': 30,
        'lr': 0.0003,
        'gamma': 0.99,
        'lamda': 0.95,
        'ratio_clip': 0.25,
        'batch_size': 256,
        'temperature_coef': 0.1,
        'device': 'cuda',
        'result_path': '/home/xukang/Project/state_filtration_for_qd/results_for_diayn/'
    }
    
    for env_name in ['Walker', 'Hopper', 'HalfCheetah']:
        for seed in [10, 20, 30]:
            config['env_config']['env_name'] = env_name
            config['seed'] = seed
            main(config, '')

    #demo('/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/Walker-missing_coord_2_3-10/','best')
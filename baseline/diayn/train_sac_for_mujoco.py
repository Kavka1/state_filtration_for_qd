from typing import List, Dict, Tuple
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from state_filtration_for_qd.utils import confirm_path_exist, make_exp_path
from state_filtration_for_qd.logger import Logger
from state_filtration_for_qd.env.common import call_env
from state_filtration_for_qd.baseline.diayn.agent_sac import DIAYN_SAC
from state_filtration_for_qd.model.latent_policy import Latent_DiagGaussianPolicy
from state_filtration_for_qd.buffer import Buffer

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
    # make loggers
    logger = Logger()
    tb = SummaryWriter(config['exp_path'])
       
    # build agent
    agent = DIAYN_SAC(config)
    buffer = Buffer(config['buffer_size'])
    # 
    z_dim   =  agent.z_dim
    p_z     =  np.full(z_dim, 1/z_dim)

    total_step, total_episode = 0, 0
    best_score = -100

    obs             =   env.reset()
    z               =   np.random.choice(z_dim, p=p_z)
    while total_step <= config['max_timesteps']:
        one_hot_z       =   np.zeros(z_dim).astype(np.float64)
        one_hot_z[z]    =   1
        obs_z           =   np.concatenate([obs, one_hot_z], -1)
        action          =   agent.choose_action(obs_z, True)
        
        obs_, reward, done, info    =    env.step(action)

        buffer.store((obs, action, reward, done, obs_, z))
        logger_dict = agent.train(buffer)

        if done:
            total_episode   +=  1
            obs             =   env.reset()
            z               =   np.random.choice(z_dim, p=p_z)
        else:
            obs             =   obs_

        if total_step % config['eval_interval'] == 0:
            rewards_across_skill = agent.evaluate(env, config['eval_episode'])
            max_skill_score = np.max(rewards_across_skill)
            mean_skill_score = np.mean(rewards_across_skill)
            min_skill_score = np.min(rewards_across_skill)

            if max_skill_score > best_score:
                agent.save_policy('best')
                best_score = max_skill_score

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
            print("DIAYN | Step {} | Episode {} | {}".format(total_step, total_episode, logger))
            # Log the results in tb
            tb.add_scalar('Eval/eval_return', max_skill_score, total_step)
            tb.add_scalar('Eval/max_skill_score', max_skill_score, total_step)
            tb.add_scalar('Eval/mean_skill_score', mean_skill_score, total_step)
            tb.add_scalar('Eval/min_skill_score', min_skill_score, total_step)
            for name, value_seq in list(logger.data.items()):
                tb.add_scalar(f'Train/{name}', value_seq[-1], total_step)
        
        # save models periodically
        if total_step % config['save_interval'] == 0:
            agent.save_policy(f'{total_step}')
            agent.save_discriminator(f'{total_step}')

        total_step += 1
        
    agent.save_policy('final')
    agent.save_discriminator('final')


def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    z_dim =  config['model_config']['z_dim']
    policy = Latent_DiagGaussianPolicy(
        config['model_config']['o_dim'],
        config['model_config']['a_dim'],
        config['model_config']['z_dim'],
        config['model_config']['policy_hidden_layers'],
        config['model_config']['policy_logstd_min'],
        config['model_config']['policy_logstd_max'],
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
                env.render()
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
            'policy_hidden_layers': [256, 256],
            'value_hidden_layers': [256, 256],
            'disc_hidden_layers': [256, 256],
            'policy_logstd_min': -20,
            'policy_logstd_max': 2,
        },
        'env_config': {
            'env_name': 'HalfCheetah',
            'missing_obs_info': {
                'missing_joint':    [],
                'missing_coord':    [],
                'missing_leg':      []
            }
        },

        'max_timesteps': 1000000,
        'buffer_size': 1000000,
        'eval_interval': 20000,
        'save_interval': 100000,
        'eval_episode': 1,

        'reward_tradeoff_ex': 1,
        'reward_tradeoff_in': 0.001,

        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.005,
        'fix_alpha': 0.1,
        'batch_size': 256,
        'train_policy_delay': 2,
        'device': 'cuda',
        'result_path': '/home/xukang/Project/state_filtration_for_qd/results_for_diayn/'
    }
    
    for seed in [40, 50]:
        for env_config in [
            {
                'env_name': 'Minitaur',
                'missing_obs_info': {
                    'missing_angle':    [],
                }
            }
        ]:    
            config['env_config'] = env_config
            config['seed'] = seed
            main(config, 'r_ex-10_skill')

    #demo('/home/xukang/Project/state_filtration_for_qd/results_for_diayn/r_ex-Ant-10/','final')
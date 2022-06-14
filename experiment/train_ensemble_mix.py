from typing import List, Dict, Tuple
import numpy as np
import torch
import yaml
import os
import cv2
from torch.utils.tensorboard import SummaryWriter

from state_filtration_for_qd.utils import confirm_path_exist, make_exp_path
from state_filtration_for_qd.agent.ensemble_mix import Ensemble
from state_filtration_for_qd.model.flat_policy import FixStdGaussianPolicy
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
    ensemble = Ensemble(config)
    num_primitive = ensemble.num_primitive
    # make loggers
    logger = Logger()
    tb = SummaryWriter(config['exp_path'])

    total_step, total_episode, total_iteration = 0, 0, 0
    best_score = -100
    
    while total_step <= config['max_timesteps']:
        logger_dict = ensemble.rollout_update()

        total_step = ensemble.total_step
        total_episode = ensemble.total_episode
        
        reward_across_primitives = ensemble.evaluate(env, config['eval_episode'])
        max_primitive_score      = np.max(reward_across_primitives)
        mean_primitive_score     = np.mean(reward_across_primitives)
        min_primitive_score      = np.min(reward_across_primitives)
        if max_primitive_score > best_score:
            ensemble.save_policy('best')
            best_score = max_primitive_score
            
        logger_dict.update({
            'iteration': total_iteration,
            'episode': total_episode, 
            'steps': total_step, 
            'max_primitive_score': max_primitive_score,
            'mean_primitive_score': mean_primitive_score,
            'min_primitive_score': min_primitive_score
        })
        logger.store(**logger_dict)    
        # Log the results in terminal
        print("| Iteration {} | Step {} | {}".format(total_iteration, total_step, logger))
        # Log the evaluation results in tb
        tb.add_scalar('Eval/max_primitive_score', max_primitive_score, total_iteration)
        tb.add_scalar('Eval/min_primitive_score', min_primitive_score, total_iteration)
        tb.add_scalar('Eval/mean_primitive_score', mean_primitive_score, total_iteration)
        tb.add_scalar('Eval/total_step', total_step, total_iteration)
        # Log the information of the training process in tb
        for name, value_seq in list(logger.data.items()):
            tb.add_scalar(f'Train/{name}', value_seq[-1], total_step)
        
        # save log
        if total_iteration % config['log_interval'] == 0:
            logger.save(config['exp_path'] + 'log.pkl')

        # save models periodically
        if total_iteration % config['save_interval'] == 0:
            ensemble.save_policy(f'{total_iteration}')
            ensemble.save_inverse_model(f'{total_iteration}')

        total_iteration += 1

    ensemble.save_policy('final')
    ensemble.save_inverse_model('final')

        


def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    num_primitive = config['num_primitive']
    all_policy = []

    for k in range(num_primitive):
        policy = FixStdGaussianPolicy(
            config['model_config']['o_dim'],
            config['model_config']['a_dim'],
            config['model_config']['policy_hidden_layers'],
            config['model_config']['action_std'],
            config['model_config']['policy_activation']
        )
        policy.load_model(path + f'model/policy_{k}_{remark}')
        all_policy.append(policy)
    
    env = call_env(config['env_config'], is_render=True)
    
    if config['env_config']['env_name'] == 'Quadruped':
        is_dmc = True
    else:
        is_dmc = False
    
    for _ in range(100):
        for k in range(len(all_policy)):
            if is_dmc:
                video_path = '/home/xukang/Project/state_filtration_for_qd/video.avi'
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (600,480))

            policy = all_policy[k]
            done = False
            episode_r = 0
            obs = env.reset()
            while not done:
                
                if is_dmc:
                    rgb_arr = env.render()
                    video.write(cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB))
                else:
                    env.render()

                obs = torch.from_numpy(obs).float()
                a = policy.act(obs, False).detach().numpy()
                obs, r, done, info = env.step(a)
                episode_r += r
            print(f"Primitive {k} CheckPoint {remark} Episode {_} Episode Reward {episode_r}")

            if is_dmc:
                video.release()
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if not ret:
                        break
                    cv2.imshow('Playback', frame)
                    
                cap.release()
                
                cv2.destroyAllWindows()
                os.remove(video_path)



if __name__ == '__main__':
    config = {
        'seed': 10,

        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'policy_hidden_layers': [64, 64],
            'value_hidden_layers': [128, 128],
            'idm_hidden_layers': [128, 128],
            'action_std': 0.4,
            'idm_logstd_min': -10,
            'idm_logstd_max': 0.5,
            'policy_activation': 'Tanh'
        },
        'env_config': {
            'env_name': 'Minitaur',
            'missing_obs_info': {
                'missing_angle': ['1', '2', '3', '4'],
            }
        },

        'num_primitive': 10,

        'reward_tradeoff': 0.001,

        'max_timesteps': 20000000,
        'save_interval': 40,
        'log_interval': 10,
        'eval_episode': 1,
        
        'num_workers': 10,
        'num_worker_rollout': 2,
        'num_epoch': 10,
        'lr': 0.0003,
        'gamma': 0.99,
        'lamda': 0.95,
        'ratio_clip': 0.25,
        'batch_size': 256,
        'temperature_coef': 0.1,
        'device': 'cuda',
        'result_path': '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble_mix/'
    }
    

    for env in [
        'Hopper',
        #'Walker',
        #'Ant',
        #'Minitaur'
    ]:
        if env == 'Hopper':
            env_config = {
                'env_name': 'Hopper',
                'missing_obs_info': {
                    'missing_coord':    [],
                    'missing_joint':    [],
                    'missing_leg':      ['1']
                }
            }
            tradeoff = 0.05
        elif env == 'Walker':
            env_config = {
                'env_name': 'Walker',
                'missing_obs_info': {
                    'missing_coord':    [],
                    'missing_joint':    [],
                    'missing_leg':      ['1']
                }
            }
            tradeoff = 0.05
        elif env == 'Ant':
            env_config = {
                'env_name': 'Ant',
                'missing_obs_info': {
                    'missing_coord':    [],
                    'missing_joint':    [],
                    'missing_leg':      ['1','2','3','4']
                }
            }
            tradeoff = 0.01
        elif env == 'Minitaur':
            env_config = {
                'env_name': 'Minitaur',
                'missing_obs_info': {
                    'missing_angle':    ['1','2','3','4'],
                }
            }
            tradeoff = 0.001
        else:
            raise ValueError


        for seed in [10, 20, 30, 40, 50]:
            config['env_config'] = env_config
            config['reward_tradeoff'] = tradeoff
            config['seed'] = seed
            #main(config, '')

    demo('/home/xukang/Project/state_filtration_for_qd/results_for_ensemble_mix/Walker-missing_leg_1-20/','final')
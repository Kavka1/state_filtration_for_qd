from typing import List, Dict, Tuple
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from state_filtration_for_qd.utils import confirm_path_exist, make_exp_path
from state_filtration_for_qd.agent.ensemble import Ensemble
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
    local_iteration = 0         # iteration count for each primitive
    best_score_across_primitive = [0] * num_primitive
    for k in range(num_primitive):
        local_iteration = 0
        while total_step <= config['max_timesteps_per_primitive']:
            logger_dict = ensemble.rollout_update(k)

            total_step = ensemble.total_steps
            total_episode = ensemble.total_episodes

            reward_current_primitive = ensemble.evaluate(env, config['eval_episode'])
            if reward_current_primitive > best_score_across_primitive[k]:
                ensemble.save_policy(k, 'best')
                best_score_across_primitive[k] = reward_current_primitive
            
            logger_dict.update({
                'iteration': total_iteration,
                'episode': total_episode, 
                'steps': total_step, 
                'current_primitive_return': reward_current_primitive,
            })
            logger_dict.update({f'primitive_{j}_score': 0 for j in range(num_primitive)})
            logger_dict[f'primitive_{k}_score'] = reward_current_primitive

            logger.store(**logger_dict)
            logger.save(config['exp_path'] + 'log.pkl')
            # Log the results in terminal
            print("| Iteration {} | Step {} | {}".format(total_iteration, total_step, logger))
            # Log the evaluation results in tb
            tb.add_scalar('Eval/current_primitive_return', reward_current_primitive, total_iteration)
            tb.add_scalar('Eval/total_step', total_step, total_iteration)
            for j in range(num_primitive):
                tb.add_scalar(f'Eval/primitive_{j}_score', logger_dict[f'primitive_{j}_score'], total_iteration)
            # Log the information of the training process in tb
            for name, value_seq in list(logger.data.items()):
                tb.add_scalar(f'Train/{name}', value_seq[-1], total_step)
            # save models periodically
            if total_iteration % config['save_interval'] == 0:
                ensemble.save_policy(k, f'{local_iteration}')
                ensemble.save_inverse_model(k, f'{local_iteration}')

            total_iteration += 1
            local_iteration += 1

def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    all_policy = []
    for k in config['num_primitive']:
        policy = FixStdGaussianPolicy(
            config['model_config']['o_dim'],
            config['model_config']['a_dim'],
            config['model_config']['policy_hidden_layers'],
            config['model_config']['action_std'],
        )
        policy.load_model(path + f'model/policy_{k}_{remark}')
        all_policy.append(policy)
    
    env = call_env(config['env_config'])
    
    for _ in range(100):
        for k in range(len(all_policy)):
            policy = all_policy[k]
            done = False
            episode_r = 0
            obs = env.reset()
            while not done:
                env.render()
                obs = torch.from_numpy(obs).float()
                a = policy.act(obs, False).detach().numpy()
                obs, r, done, info = env.step(a)
                episode_r += r
            print(f"Primitive {k} CheckPoint {remark} Episode {_} Episode Reward {episode_r}")


if __name__ == '__main__':
    config = {
        'seed': 10,

        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'policy_hidden_layers': [64, 64],
            'value_hidden_layers': [128, 128],
            'idm_hidden_layers': [256, 256],
            'action_std': 0.4,
            'idm_logstd_min': -10,
            'idm_logstd_max': 0.5
        },
        'env_config': {
            'env_name': 'Walker',
            'missing_obs_info': {
                'missing_joint': ['foot', 'leg', 'thigh'],
                'missing_coord': []
            }
        },

        'num_primitive': 10,
        'max_timesteps_per_primitive': 2000000,
        'save_interval': 50,
        'eval_episode': 10,
        
        'num_workers': 10,
        'num_worker_rollout': 5,
        'reward_tradeoff': 0.01,
        'num_epoch': 20,
        'lr': 0.0003,
        'gamma': 0.99,
        'lamda': 0.95,
        'ratio_clip': 0.25,
        'batch_size': 256,
        'temperature_coef': 0.1,
        'device': 'cpu',
        'result_path': '/home/xukang/Project/state_filtration_for_qd/results_for_ensemble/'
    }
    
    for seed in [10, 20, 30]:
        config['seed'] = seed
        main(config, 'tradeoff_0.01')

    #demo('/home/xukang/Project/state_filtration_for_qd/results/tradeoff_0.01-Walker-missing_coord_2_3-10/','best')
from copy import copy
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


def plot(csv_path: str, title: str) -> None:
    new_df = []

    for path in csv_path:
        seed = path.split('.')[0].split('-')[-1]
        if 'Hopper' in path:
            env = 'Hopper'
        elif 'Walker' in path:
            env = 'Walker'
        elif 'Ant' in path:
            env = 'Ant'
            
        with open(path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(path)

        if '/ensemble/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            baseline_rewards        =   primitive_scores[:,0]
            # process the data
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['DiR'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/single/' in path:
            primitive_scores = df.values[:,:]
            baseline_rewards        =   primitive_scores[:,0]
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : baseline_rewards,
                    'alg'                       : ['Single'] * len(baseline_rewards),
                    'seed'                      : [seed]  * len(baseline_rewards)
                })
            )
        elif '/diayn_ppo/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['DIAYN'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/smerl_ppo/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['SMERL'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/dvd/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['DvD'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/multi/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            new_df.append(
                pd.DataFrame({
                    'env'                       : [env] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['Multi'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 


    new_df = pd.concat(new_df)

    # plot
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    sns.barplot(
        data    =   new_df,
        x       =   'env',
        y       =   'return',
        hue     =   'alg',
        #style   =   'alg',
        ax      =   ax
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('Env', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.set_title(title, fontsize=11)

    plt.show()



if __name__ == '__main__':
    env_and_deffective_sensot = [
        ('Hopper', 'leg_1'),
        ('Walker', 'leg_1'),
        ('Ant',    'coord_2')
    ]
    
    all_paths = []
    for env_deff in env_and_deffective_sensot:
        env = env_deff[0]
        deff = env_deff[1]

        for alg in ['ensemble', 'dvd', 'smerl_ppo', 'multi',]:
            for seed in ['10','20', '30','40','50','60','70','80']:
                all_paths.append(
                    f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/{env}-defective_sensor-{deff}-{seed}.csv'
                )

    plot(
        all_paths,
        f'Deffective sensors'
    )
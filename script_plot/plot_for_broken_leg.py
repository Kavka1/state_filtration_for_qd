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
        if 'leg' in path:
            broken_pos = 'leg'
        elif 'foot' in path:
            broken_pos = 'foot'
        elif 'ankle' in path:
            broken_pos = 'ankle'
        elif 'hip' in path:
            broken_pos = 'hip'
            
        with open(path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(path)

        if '/ensemble/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            baseline_rewards        =   primitive_scores[:,0]
            # process the data
            new_df.append(
                pd.DataFrame({
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['Ensemble Iterative'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/ensemble_mix/' in path:
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            baseline_rewards        =   primitive_scores[:,0]
            # process the data
            new_df.append(
                pd.DataFrame({
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['Ensemble Mix'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
        elif '/single/' in path:
            primitive_scores = df.values[:,:]
            baseline_rewards        =   primitive_scores[:,0]
            new_df.append(
                pd.DataFrame({
                    'broken'                    : [broken_pos] * len(baseline_rewards),
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
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
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
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
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
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
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
                    'broken'                    : [broken_pos] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['Multiple'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 


    new_df = pd.concat(new_df)

    # plot
    sns.set_style('white')
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    sns.barplot(
        data    =   new_df,
        x       =   'broken',
        y       =   'return',
        hue     =   'alg',
        #style   =   'alg',
        ax      =   ax
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('Damage Joint', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.set_title(title, fontsize=11)

    plt.show()



if __name__ == '__main__':
    env = 'Walker'
    broken_a = 'right_leg'
    broken_b = 'right_foot'
    plot(
        [
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_a}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_a}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_a}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_a}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_a}-50.csv',
            
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_a}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_a}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_a}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_a}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_a}-50.csv',
            
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_a}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_a}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_a}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_a}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_a}-50.csv',

            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_a}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_a}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_a}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_a}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_a}-50.csv',

            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_a}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_a}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_a}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_a}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_a}-50.csv',


            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_b}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_b}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_b}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_b}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_broken_{broken_b}-50.csv',
            
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_b}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_b}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_b}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_b}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/dvd/{env}_broken_{broken_b}-50.csv',

            
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_b}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_b}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_b}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_b}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/smerl_ppo/{env}_broken_{broken_b}-50.csv',

            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_b}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_b}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_b}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_b}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/multi/{env}_broken_{broken_b}-50.csv',

            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_b}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_b}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_b}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_b}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/single/{env}_broken_{broken_b}-50.csv',
        ],
        
        f'{env} - Damage at different joints of the leg'
    )
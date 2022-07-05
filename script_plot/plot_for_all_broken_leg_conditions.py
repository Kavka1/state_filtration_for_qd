from copy import copy
from re import L
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


csv2alg = {
    '/ensemble/': 'DiR',
    '/single/':     'PG',
    '/smerl_ppo/': 'SMERL',
    '/dvd/': 'DvD',
    '/multi/': 'Multi'
}

env_and_broken = [
    ['Hopper', 'leg', 'foot'],
    ['Walker', 'right_leg', 'right_foot'],
    ['Ant', 'ankle', 'hip'],
    ['Minitaur', 'leg_2']
]


def collect_csv_path(env: str, all_broken: List[str]) -> List[str]:
    all_paths = []
    for broken in all_broken:
        for alg in ['ensemble', 'dvd', 'smerl_ppo', 'multi', 'single']:
            for seed in ['10','20', '30','40','50','60','70','80']:
                if env == 'Minitaur':
                    if alg == 'ensemble':
                        all_paths.append(
                            f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/new_trdeoff-{env}-missing_angle_1_2_3_4-damage_{broken}-{seed}.csv'
                        )
                    elif alg == 'single':
                        all_paths.append(
                            f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/new_trdeoff-{env}_damage_{broken}-{seed}.csv'
                        )
                    else:
                        all_paths.append(
                            f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/{env}_damage_{broken}-{seed}.csv'
                        )
                else:
                    all_paths.append(
                        f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/{env}_broken_{broken}-{seed}.csv'
                    )
    return all_paths


def plot(title: str) -> None:
    sns.set_style('white')
    fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout = True, figsize= (7, 5),)
    for i, ax in enumerate(axs.flat):

        env = env_and_broken[i][0]
        all_broken = env_and_broken[i][1:]
        all_path = collect_csv_path(env, all_broken)

        new_df = []
        for path in all_path:
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

            for csvkey in list(csv2alg.keys()):
                if csvkey in path:
                    alg_key = csvkey
                    break
            primitive_scores = df.values[:,:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            baseline_rewards        =   primitive_scores[:,0]
            # process the data
            new_df.append(
                pd.DataFrame({
                    'broken'                    : [f'broken {broken_pos}'] * len(max_primitive_rewards),
                    'return'                    : max_primitive_rewards,
                    'alg'                       : [csv2alg[alg_key]] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 


        new_df = pd.concat(new_df)

        # plot
        sns.barplot(
            data    =   new_df,
            x       =   'broken',
            y       =   'return',
            hue     =   'alg',
            #style   =   'alg',
            ax      =   ax,
            capsize =   0.04,
            linewidth=  1.2,
            edgecolor = '1.'
        )
        
        #ax.set_ylim([1000, 4000])
        if i != 0:
            ax.legend().remove()
        else:
            ax.legend().set_title('')
            #sns.move_legend(ax, 'upper left', bbox_to_anchor=(.9, .1), ncol=1)

        if env == 'Hopper':
            ax.set_xticklabels(['Broken leg', 'Broken foot'], fontsize=12)
        elif env == 'Walker':
            ax.set_xticklabels(['Broken leg', 'Broken foot'], fontsize=12)
        elif env == 'Ant':
            ax.set_xticklabels(['Broken ankle', 'Broken hip'], fontsize=12)
        else:
            ax.set_xticklabels(['Motor failure'], fontsize=12)

        ax.set_xlabel('', fontsize=12)
        if i in [1, 3]:
            ax.set_ylabel('', fontsize=12)
        else:
            ax.set_ylabel('Return', fontsize=12)
        ax.set_title(env, fontsize=12)
        sns.despine(fig, ax)

    plt.show()



if __name__ == '__main__':
    plot(
        'Damage at different joints of the leg'
    )
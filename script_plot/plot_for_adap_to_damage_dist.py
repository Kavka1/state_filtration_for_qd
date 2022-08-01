from copy import copy
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


all_seeds = [f'{int(n+1)*10}' for n in range(8)]
all_algs = [
    'DiR',
    'SMERL',
    'DvD',
    'Multi'
]
all_alg_path_name = [
    'ensemble',
    'smerl_ppo',
    'dvd',
    'multi'
]
num_primitive = 10


def plot(title: str, broken_name: str) -> None:
    new_df = []

    def get_csv_path(alg_path_name: str, broken_name: str, seed: int) -> str:
        if 'Minitaur' in broken_name and alg_path_name == 'ensemble':
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg_path_name}/new_trdeoff-Minitaur-missing_angle_1_2_3_4-damage_leg_2-{seed}.csv'
        else:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg_path_name}/{broken_name}-{seed}.csv'
        return csv_path


    for alg, alg_path_name in zip(all_algs, all_alg_path_name):
        for seed in all_seeds:
            csv_path = get_csv_path(alg_path_name, broken_name, seed)
            df_values = pd.read_csv(csv_path).values.tolist()[0]
            df_values.sort()
            new_df.append(
                pd.DataFrame({
                    'seed':             [seed for _ in range(num_primitive)],
                    'ret':              df_values,
                    'primitive':        [f'Policy {p+1}' for p in range(num_primitive)],
                    'alg':              [f'{alg}' for _ in range(num_primitive)]
                })
            )

    new_df = pd.concat(new_df)

    # plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14,4))

    sns.barplot(
        data    =   new_df,
        x       =   'primitive',
        y       =   'ret',
        hue     =   'alg',
        ci      =   'sd',
        saturation  = 0.8,
        #linewidth   = 1.,
        #edgecolor = '.5',
        ax      =   ax
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title(title, fontsize=12)

    plt.show()



if __name__ == '__main__':
    plot(
        f'Adaptation Performance in Minitaur - motor failure',
        'Minitaur_damage_leg_2'
    )
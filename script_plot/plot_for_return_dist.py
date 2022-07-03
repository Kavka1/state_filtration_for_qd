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
all_envs = [
    'Hopper',
    'Walker',
    'Ant',
    'Minitaur'
]
num_primitive = 10


def plot(title: str) -> None:
    new_df = []

    for env in all_envs:
        csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env}_return_dist.csv'
        df = pd.read_csv(csv_path)
        max_ret = np.max(df.values)

        for seed in all_seeds:
            ret_of_all_primitive = list(df[f'seed {seed}'])
            ret_of_all_primitive.sort()
            normalized_ret = [ret / max_ret for ret in ret_of_all_primitive]

            new_df.append(
                pd.DataFrame({
                    'normalized ret':   normalized_ret,
                    'primitive':        [f'policy {p+1}' for p in range(num_primitive)],
                    'env':              [f'{env}' for _ in range(len(normalized_ret))]
                })
            )

    new_df = pd.concat(new_df)

    # plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(11,4))

    sns.barplot(
        data    =   new_df,
        x       =   'primitive',
        y       =   'normalized ret',
        hue     =   'env',
        ci      =   'sd',
        saturation  = 0.8,
        #linewidth   = 1.,
        #edgecolor = '.5',
        ax      =   ax
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('', fontsize=11)
    ax.set_ylabel('Normalized Return', fontsize=11)
    ax.set_title(title, fontsize=11)

    plt.show()



if __name__ == '__main__':
    plot(
        f'Distribution of the normalized return across all environments'
    )
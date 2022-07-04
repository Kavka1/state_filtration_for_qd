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
all_algs = [
    'DiR',
    'DvD',
    'SMERL',
    'Multi'
]
all_alg_csv = {
    'DiR':  'ensemble',
    'DvD':  'dvd',
    'SMERL':    'smerl_ppo',
    'Multi':    'multi'
}

num_primitive = 10


def plot() -> None:
    fig, axs = plt.subplots(nrows=4,ncols=1,tight_layout=True,figsize=(6, 6))
    for i, env in enumerate(all_envs):
        ax = axs[i]
        all_df = [] 
        for alg in all_algs:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/{env}_return_dist.csv'
            
            df = pd.read_csv(csv_path)
            max_ret = np.max(df.values, axis=1)
            mean_ret = np.mean(df.values, axis=1)
            min_ret = np.min(df.values, axis=1)

            all_df += [
                pd.DataFrame({
                    'ret': max_ret,
                    'type': ['max'] * len(max_ret),
                    'alg':  [alg] * len(max_ret)
                }),
                pd.DataFrame({
                    'ret': mean_ret,
                    'type': ['mean'] * len(mean_ret),
                    'alg':  [alg] * len(mean_ret)
                }),
                pd.DataFrame({
                    'ret': min_ret,
                    'type': ['min'] * len(min_ret),
                    'alg':  [alg] * len(min_ret)
                }),
            ]

        sns.barplot(
            data= pd.concat(all_df),
            x=  'type',
            y=  'ret',
            hue= 'alg',
            ci= 'sd',
            ax=ax,
        )

        #ax.set_ylim([1000, 4000])
        ax.legend().set_title('')
        ax.set_xlabel('', fontsize=11)
        ax.set_ylabel('Return', fontsize=11)
        ax.set_title(env, fontsize=11)

    plt.show()



if __name__ == '__main__':
    plot()
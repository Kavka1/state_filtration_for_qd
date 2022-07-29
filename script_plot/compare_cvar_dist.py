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
    #'Minitaur'
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
    fig, axs = plt.subplots(nrows=3,ncols=1,tight_layout=True,figsize=(6, 6))
    for i, env in enumerate(all_envs):
        ax = axs[i]
        all_df = [] 
        for alg in all_algs:
            
            if env == 'Minitaur' and alg == 'DiR':
                csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/new_trdeoff-{env}_cvar25_dist.csv'
            else:
                csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/{env}_cvar25_dist.csv'
            
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



def plot_mean_and_min_ret_dist() -> None:
    sns.set_style('white')

    def get_csv_path(env: str, alg: str) -> str:
        if env == 'Minitaur' and alg == 'DiR':
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/new_trdeoff-{env}_cvar25_dist.csv'
        else:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/{env}_cvar25_dist.csv'
        return csv_path

    fig, axs = plt.subplots(nrows=2,ncols=1,tight_layout=True,figsize=(6.5, 4.5), sharex=True)
    
    for i, ax in enumerate(axs):
        if i == 0:
            # plot mean ret dist
            all_df = []
            for env in all_envs:
                # search the max return 
                max_ret = 1
                for alg in all_algs:
                    csv_path = get_csv_path(env, alg)
                    df = pd.read_csv(csv_path)
                    mean_ret = np.mean(df.values, axis=1)
                    max_ret = np.max(mean_ret) if np.max(mean_ret) > max_ret else max_ret

                for alg in all_algs:
                    csv_path = get_csv_path(env, alg)
                    df = pd.read_csv(csv_path)
                
                    mean_ret = np.mean(df.values, axis=1)
                    all_df.append(
                            pd.DataFrame({
                            'ret': mean_ret / max_ret,
                            'env': [env] * len(mean_ret),
                            'alg':  [alg] * len(mean_ret)
                        }),
                    )

            sns.despine(fig, axs[i])
            sns.barplot(
                data= pd.concat(all_df),
                x=  'env',
                y=  'ret',
                hue= 'alg',
                ci= 'sd',
                ax=ax,
            )
            ax.legend().set_title('')
            ax.set_xlabel('', fontsize=11)
            ax.set_ylabel('Normalized mean cvar25', fontsize=11)
            ax.set_title('', fontsize=11)
            for _, s in ax.spines.items():
                s.set_linewidth(1.1)

            #sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=4)

        else:
            # plot min ret dist
            all_df = []
            for env in all_envs:
                # search the max return 
                max_ret = -10
                for alg in all_algs:
                    csv_path = get_csv_path(env, alg)
                    df = pd.read_csv(csv_path)
                    min_ret = np.min(df.values, axis=1)
                    max_ret = np.max(min_ret) if np.max(min_ret) > max_ret else max_ret

                for alg in all_algs:
                    csv_path = get_csv_path(env, alg)
                    df = pd.read_csv(csv_path)
                
                    min_ret = np.min(df.values, axis=1)
                    all_df.append(
                            pd.DataFrame({
                            'ret': min_ret / max_ret,
                            'env': [env] * len(mean_ret),
                            'alg':  [alg] * len(mean_ret)
                        }),
                    )

            sns.despine(fig, axs[i])
            sns.barplot(
                data= pd.concat(all_df),
                x=  'env',
                y=  'ret',
                hue= 'alg',
                ci= 'sd',
                ax=ax,
            )
            ax.legend().remove()
            ax.set_xlabel('', fontsize=11)
            ax.set_ylabel('Normalized min cvar25', fontsize=11)
            ax.set_title('', fontsize=11)
            for _, s in ax.spines.items():
                s.set_linewidth(1.1)


    plt.show()


if __name__ == '__main__':
    #plot()
    plot_mean_and_min_ret_dist()
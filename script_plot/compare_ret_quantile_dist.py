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



def plot_ret_quantile() -> None:
    sns.set_style('whitegrid')

    def get_csv_path(env: str, alg: str) -> str:
        if env == 'Minitaur' and alg == 'DiR':
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/new_trdeoff-{env}_return_dist.csv'
        else:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{all_alg_csv[alg]}/{env}_return_dist.csv'
        return csv_path

    fig, axs = plt.subplots(nrows=2,ncols=2,tight_layout=True,figsize=(6, 5), sharex=False)
    
    for i, ax in enumerate(axs.flat):
        # plot mean ret dist
        all_df = []
        env = all_envs[i]
        
        max_ret = 1
        for alg in all_algs:
            csv_path = get_csv_path(env, alg)
            df = pd.read_csv(csv_path)
            max_df_ret = np.max(df.values)
            max_ret = max_df_ret if max_df_ret > max_ret else max_ret


        for alg in all_algs:
            csv_path = get_csv_path(env, alg)
            df = pd.read_csv(csv_path)

            df_values = df.values
            df_values = np.sort(df_values, axis=0)      # sort accordding to the policy performance

            for j in range(num_primitive):
                all_df.append(pd.DataFrame({
                    'ret':  df_values[j, :],
                    'quantile': [j * 10] * len(all_seeds),
                    'alg':  [alg] * len(all_seeds)
                }))

        sns.despine(fig, ax)
        sns.lineplot(
            data= pd.concat(all_df),
            x=  'quantile',
            y=  'ret',
            hue= 'alg',
            ci= 'sd',
            err_style='band',
            ax=ax,
        )

        if i == 0:
            ax.legend().set_title('')
        else:
            ax.legend().remove()
    
        ax.set_xlabel('Percentile', fontsize=12)
        if i == 1 or i == 3:
            ax.set_ylabel('', fontsize=12)
        else:
            ax.set_ylabel('Return', fontsize=12)
        ax.set_title(f'{env}', fontsize=12)
        
        for _, s in ax.spines.items():
            s.set_linewidth(1.2)
        #for _, s in ax.spines.items():
        #   s.set_linewidth(1.1)

        #sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=4)

    plt.show()


if __name__ == '__main__':
    #plot()
    plot_ret_quantile()
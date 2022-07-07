from copy import copy
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=[
    '#4E79A7', '#F28E2B', '#E15759', 'dimgrey','#59A14E',
    '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


csv2alg = {
    '/ensemble/': 'DiR',
    '/single/':     'PG',
    '/smerl_ppo/': 'SMERL',
    '/dvd/': 'DvD',
    '/multi/': 'Multi'
}

env_and_param = [
    ['Hopper', 'mass'],
    ['Hopper', 'fric'],
    ['Walker', 'mass'],
    ['Walker', 'fric'],
    ['Ant', 'mass'],
    ['Ant', 'fric']
]

def collect_csv_path(env: str, param: str) -> List[str]:
    all_paths = []
    for alg in ['ensemble', 'dvd', 'smerl_ppo', 'multi', 'single']:
        for seed in ['10','20', '30','40','50','60','70','80']:
            all_paths.append(
                f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/{env}_dynamics_{param}-{seed}.csv'
            )
    return all_paths

def title_fig(env: str, param: str) -> str:
    if env in ['Hopper', 'Walker']:
        if param == 'fric':
            title = f'{env} - foot friction'
        else:
            title = f'{env} - foot {param}' 
    else:
        if param == 'mass':
            title = 'Ant - leg mass'
        else:
            title = 'Ant - ankle friction'
    return title


def plot() -> None:
    sns.set_style('white')
    fig, axes = plt.subplots(nrows=3, ncols=2, tight_layout=True, figsize=(9, 9.5), sharey=False, sharex=True)
    for i, ax in enumerate(axes.flat):

        env = env_and_param[i][0]
        param = env_and_param[i][1]
        all_path = collect_csv_path(env, param)

        new_df = []
        for path in all_path:
            seed = path.split('.')[0].split('-')[-1]
            with open(path, 'r', encoding='utf-8') as f:
                df = pd.read_csv(path)

            for csvkey in list(csv2alg.keys()):
                if csvkey in path:
                    alg_key = csvkey
                    break
            primitive_scores = df.values[:,1:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1).tolist()
            
            # remove half data
            all_param_scale = df.values[:,0].tolist()
            for j in reversed(range(len(max_primitive_rewards))):
                if j%2 == 0:
                    temp = copy(max_primitive_rewards[j])
                    scale_temp = copy(all_param_scale[j])

                    max_primitive_rewards.remove(temp)
                    all_param_scale.remove(scale_temp)


            # process the data
            new_df.append(
                pd.DataFrame({
                    'param scale'               : all_param_scale,
                    'return'                    : max_primitive_rewards,
                    'alg'                       : [f'{csv2alg[alg_key]}' for _ in range(len(max_primitive_rewards))],
                    'seed'                      : [f'{seed}' for _ in range(len(max_primitive_rewards))]
                })
            ) 

        new_df = pd.concat(new_df)

        # plot
        sns.lineplot(
            data    =   new_df,
            x       =   'param scale',
            y       =   'return',
            hue     =   'alg',
            style   =   'alg',
            ax      =   ax,
            dashes  =   True,
            markers =   True,
            err_style   =   'band',
            #sizes= (5,5,5,5,5)
            linewidth = 1
        )
        #ax.set_ylim([1000, 4000])
        if i == 1:
            ax.legend().set_title('')
        else:
            ax.legend().remove()
        ax.set_xlabel('Scale of the parameter', fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel('Return', fontsize=12)
        else:
            ax.set_ylabel('')
        ax.set_title(title_fig(env, param), fontsize=12)
        for _, s in ax.spines.items():
            s.set_linewidth(1.2)


        #sns.despine(fig, ax)

    plt.show()



if __name__ == '__main__':
    plot()
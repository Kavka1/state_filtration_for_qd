from copy import copy
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


plt.rcParams['axes.prop_cycle']  = cycler(color=[
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
    '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


def plot(csv_path: str, title: str) -> None:
    new_df = []
    for path in csv_path:
        seed = path.split('.')[0].split('-')[-1]
        with open(path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(path)

        all_param_scale = df.values[:,0]
        if 'ensemble' in path:
            primitive_scores = df.values[:,1:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            baseline_rewards        =   primitive_scores[:,0]
            # process the data
            new_df.append(
                pd.DataFrame({
                    'param scale'               : all_param_scale,
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['Ensemble'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 
            new_df.append(
                pd.DataFrame({
                    'param scale'               : all_param_scale,
                    'return'                    : baseline_rewards,
                    'alg'                       : ['Single'] * len(max_primitive_rewards),
                    'seed'                      : [seed]  * len(max_primitive_rewards)
                })
            )
        elif 'diayn' in path:
            primitive_scores = df.values[:,1:]
            max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
            new_df.append(
                pd.DataFrame({
                    'param scale'               : all_param_scale,
                    'return'                    : max_primitive_rewards,
                    'alg'                       : ['DIAYN'] * len(max_primitive_rewards),
                    'seed'                      : [seed] * len(max_primitive_rewards)
                })
            ) 

    new_df = pd.concat(new_df)

    # plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    sns.lineplot(
        data    =   new_df,
        x       =   'param scale',
        y       =   'return',
        hue     =   'alg',
        style   =   'alg',
        ax      =   ax,
        dashes  =   False,
        markers =   True,
        err_style   =   'band'
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('Dynamics parameter scale', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.set_title(title, fontsize=12)

    plt.show()



if __name__ == '__main__':
    env_name = 'Hopper'
    disturbed_param = 'mass'

    plot(
        [
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env_name}_dynamics_{disturbed_param}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env_name}_dynamics_{disturbed_param}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{env_name}_dynamics_{disturbed_param}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{env_name}_dynamics_{disturbed_param}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{env_name}_dynamics_{disturbed_param}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{env_name}_dynamics_{disturbed_param}-30.csv',
        ],
        
        f'Hopper - dynamics parameter scaled at the foot {disturbed_param}'
    )
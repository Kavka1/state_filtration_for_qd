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
    if isinstance(csv_path, List):
        new_df = []

        for path in csv_path:
            seed = path.split('.')[0].split('-')[-1]

            with open(path, 'r', encoding='utf-8') as f:
                df = pd.read_csv(path)

            chosen_noise_num = 15
            all_noise_scale = df.values[:chosen_noise_num,0]

            if 'ensemble' in path:
                primitive_scores = df.values[:chosen_noise_num,1:]
                max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
                baseline_rewards        =   primitive_scores[:,0]
                # process the data
                new_df.append(
                    pd.DataFrame({
                        'noise scale'               : all_noise_scale,
                        'return'                    : max_primitive_rewards,
                        'alg'                       : ['Ensemble Max'] * len(max_primitive_rewards),
                        'seed'                      : [seed] * len(max_primitive_rewards)
                    })
                ) 
                new_df.append(
                    pd.DataFrame({
                        'noise scale'               : all_noise_scale,
                        'return'                    : baseline_rewards,
                        'alg'                       : ['Single'] * len(max_primitive_rewards),
                        'seed'                      : [seed]  * len(max_primitive_rewards)
                    })
                )
            elif 'diayn' in path:
                primitive_scores = df.values[:chosen_noise_num,1:]
                max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
                new_df.append(
                    pd.DataFrame({
                        'noise scale'               : all_noise_scale,
                        'return'                    : max_primitive_rewards,
                        'alg'                       : ['DIAYN'] * len(max_primitive_rewards),
                        'seed'                      : [seed] * len(max_primitive_rewards)
                    })
                ) 

        new_df = pd.concat(new_df)

    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(csv_path)

        chosen_noise_num = 10
        all_noise_scale = df.values[:chosen_noise_num,0]
        max_return = np.max(df.values[:chosen_noise_num, :])

        primitive_scores = df.values[:chosen_noise_num,1:]
        max_primitive_rewards   =   np.max(primitive_scores, axis=-1)
        baseline_rewards        =   primitive_scores[:,0]

        # process the data
        new_df = [
            pd.DataFrame({
                'noise scale'               : all_noise_scale,
                'return'                    : max_primitive_rewards,
                'alg'                       : ['Ensemble Max'] * len(max_primitive_rewards)
            }),
            pd.DataFrame({
                'noise scale'               : all_noise_scale,
                'return'                    : baseline_rewards,
                'alg'                       : ['Single'] * len(max_primitive_rewards)
            })
        ]
        new_df = pd.concat(new_df)

    # plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    sns.lineplot(
        data    =   new_df,
        x       =   'noise scale',
        y       =   'return',
        hue     =   'alg',
        style   =   'alg',
        dashes  =   False,
        markers =   True,
        ax      =   ax
    )
    
    #ax.set_ylim([1000, 4000])
    ax.legend().set_title('')
    ax.set_xlabel('Observation Gaussian Noise Std', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.set_title(title, fontsize=11)

    plt.show()



if __name__ == '__main__':
    mark = 'Ant_coord_2_3_4_5'
    plot(
        [
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{mark}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{mark}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{mark}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{mark}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/{mark}-50.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{mark}-10.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{mark}-20.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{mark}-30.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{mark}-40.csv',
            f'/home/xukang/Project/state_filtration_for_qd/statistic/diayn/{mark}-50.csv',
        ],
        
        'Ant - noise at the coordinate 2 3 4 5'
    )
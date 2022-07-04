from typing import List, Dict, Tuple
import pandas as pd
import numpy as np



if __name__ == '__main__':
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]
    all_algs    = [
        'ensemble',
        'dvd',
        'smerl_ppo',
        'multi'
    ]
    all_envs = [
        'Hopper',
        'Walker',
        'Ant',
        'Minitaur'
    ]

    csv_log = {f'{env}': {f'{alg}': [] for alg in all_algs} for env in all_envs}

    for env in all_envs:
        for alg in all_algs:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/{env}_diversity_score.csv'
            df = pd.read_csv(csv_path)
            df_values = df.values
            print(f'{env}-{alg}: diversity mean - {np.mean(df_values)} diversity std - {np.std(df_values)}')

            csv_log[env][alg] += [float(np.mean(df.values)), float(np.std(df.values))]


    csv_log = pd.DataFrame(csv_log)
    csv_log.to_csv('/home/xukang/Project/state_filtration_for_qd/statistic/diversity_score.csv')
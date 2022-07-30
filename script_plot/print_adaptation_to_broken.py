from typing import List, Dict, Tuple
import pandas as pd
import numpy as np



if __name__ == '__main__':
    all_seeds = [f'{int(n+1) * 10}' for n in range(8)]
    all_algs    = [
        'multi',
        'dvd',
        'smerl_ppo',
        'ensemble',
    ]
    all_broken_components = [
        'Ant_broken_ankle',
        'Ant_broken_hip',
        'Hopper_broken_foot',
        'Hopper_broken_leg',
        'Walker_broken_right_leg',
        'Walker_broken_right_foot',
        'Minitaur_damage_leg_2'
    ]

    for broken in all_broken_components:
        for alg in all_algs:
            if 'Minitaur' in broken and alg == 'ensemble':
                broken = 'new_trdeoff-Minitaur-missing_angle_1_2_3_4-damage_leg_2'

            performance_over_all_seeds = []
            for seed in [f'{(s+1)*10}' for s in range(8)]:
                csv_path = f'/Users/xukang/Code/state_filtration_for_qd/statistic/{alg}/{broken}-{seed}.csv'
                df = pd.read_csv(csv_path)
                df_values = df.values
                performance_over_all_seeds.append(np.max(df_values))

                #csv_log[sensor][alg].append(np.max(df_values))
            mean, std = np.mean(performance_over_all_seeds), np.std(performance_over_all_seeds)

            print(f'{broken}-{alg}: {round(mean, 1)}\pm{round(std, 1)}')
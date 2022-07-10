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
    all_sensor_failure = [
        
        '2',
        '3',
        '4',
        '5'
    ]

    csv_log = {f'coord {sensor}': {f'{alg}': [] for alg in all_algs} for sensor in all_sensor_failure}

    for sensor in all_sensor_failure:
        for alg in all_algs:
            
            performance_over_all_seeds = []
            for seed in [f'{(s+1)*10}' for s in range(8)]:
                csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/{alg}/Ant-defective_sensor-coord_{sensor}-{seed}.csv'
                df = pd.read_csv(csv_path)
                df_values = df.values
                performance_over_all_seeds.append(np.max(df_values))

                #csv_log[sensor][alg].append(np.max(df_values))
            mean, std = np.mean(performance_over_all_seeds), np.std(performance_over_all_seeds)

            print(f'{sensor}-{alg}: mean - {round(mean, 2)} std - {round(std, 2)}')
            csv_log[f'coord {sensor}'][alg].append((round(mean, 2), round(std, 2)))

    csv_log = pd.DataFrame(csv_log)
    csv_log.to_csv('/home/xukang/Project/state_filtration_for_qd/statistic/defective_sensor.csv')
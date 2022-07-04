from typing import List, Dict
import pandas as pd
import numpy as np



if __name__ == '__main__':
    for seed in [f'{int(s+1)*10}' for s in range(8)]:
        for broken_leg in ['1', '2', '3', '4']:
            csv_path = f'/home/xukang/Project/state_filtration_for_qd/statistic/ensemble/new_trdeoff-Minitaur-missing_angle_1_2_3_4-damage_leg_{broken_leg}-{seed}.csv'
            df = pd.read_csv(csv_path)
            single_df = {'primitive 0': [float(df.values[0][0])]}
            single_df = pd.DataFrame(single_df)
            single_df.to_csv(f'/home/xukang/Project/state_filtration_for_qd/statistic/single/new_trdeoff-Minitaur_damage_leg_{broken_leg}-{seed}.csv', index=False)
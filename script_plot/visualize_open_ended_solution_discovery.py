from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


list_of_rewards = [
    2786.515, 1841.670, 1016.987, 2837.658, 3011.109, 3033.62, 2308.19, 3019.68, 2735.39, 2779.73
]
list_of_policy_id = list(range(1, 11))


sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
sns.despine(fig, ax)

sns.pointplot(
    x= list_of_policy_id,
    y= list_of_rewards,
    linestyles= '',
    ax= ax
)


for _, s in ax.spines.items():
    s.set_linewidth(1.5)
    s.set_color('black')
ax.set_ylim((800, 4800))
ax.set_xlabel('Policy index', fontsize=12)
ax.set_ylabel('Return', fontsize=12)
ax.set_yticklabels([1000, 1500, 2000, 2500, 3000, 3500, 4000])

plt.show()
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    embedding_1 = [float(embed.split(',')[0][1:]) for embed in df['embedding']]
    embedding_2 = [float(embed.split(',')[1][1:-1]) for embed in df['embedding']]
    model       = df['model']
    new_df = pd.DataFrame({
        'embedding_1':  embedding_1,
        'embedding_2':  embedding_2,
        'model':        model
    })

    fig, ax = plt.subplots(1,1,figsize=(7,6))
    sns.scatterplot(
        data= new_df,
        x=  'embedding_1',
        y=  'embedding_2',
        hue=  'model',
        style= 'model',
        palette= 'Set1',
        ax= ax
    )

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('t-SNE embedding', fontsize=11)

    plt.show()


if __name__ == '__main__':
    main(
        '/home/xukang/Project/state_filtration_for_qd/statistic/tsne/walker-missing_leg_1-10-obs.csv'
    )
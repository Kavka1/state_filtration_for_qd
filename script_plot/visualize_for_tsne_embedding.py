from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    embedding_1 = [embed[0] for embed in df['embedding']]
    embedding_2 = [embde[1] for embed in df['embedding']]
    model       = df['model']
    new_df = pd.DataFrame({
        'embedding_1':  embedding_1,
        'embedding_2':  embedding_2,
        'model':        model
    })

    fig, ax = plt.subplot(figsize=(5,6))
    sns.scatterplot(
        data= new_df,
        x=  'embedding_1',
        y=  'embedding_2',
        hue=  'model',
        style= 'model',
        ax= ax
    )

    ax.set_xlabel('')
    ax.set_ylaebl('')
    ax.set_title('t-SNE embedding', fontsize=11)

    plt.show()


if __name__ == '__main__':
    main(
        '/home/xukang/Porject/state_filtration_for_qd/statistic/tsne/Walker-missing_leg_1-10-policy.csv'
    )
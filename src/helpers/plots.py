import pandas as pd
import matplotlib.pyplot as plt


def barplot(data_a: pd.DataFrame, data_b: pd.DataFrame = None, title='Title'):
    num_columns = 3
    num_rows = (len(data_a.columns) + num_columns - 1) // num_columns

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(60, 45))
    axs = axs.flatten()

    for i, column in enumerate(data_a.columns):
        ax = axs[i]

        data_a[column].value_counts().sort_index().plot(kind='bar', ax=ax, color='blue', position=0,
                                                        width=0.4, label='Original dataset')

        if data_b is not None:
            data_b[column].value_counts().sort_index().plot(kind='bar', ax=ax, color='orange', position=1,
                                                            width=0.4, label='Fixed dataset')

        ax.set_title(f'{column}')
        ax.set_ylabel('Count')
        ax.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # plt.show()

    fig.savefig(title)

import matplotlib.pyplot as plt
import numpy as np


def barplot(data: dict):
    fig, ax = plt.subplots(constrained_layout=True)
    x = np.arange(len(data))  # the label locations
    width = 0.35  # the width of the bars

    for i, (attribute, measurement) in enumerate(data.items()):
        print(attribute + " ->" + str(measurement))
        rects = ax.bar(x + i * width, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=8, color='white')  # Adjusted bar_label parameters

    ax.set_ylabel('Disparate Impact')
    ax.set_title('Disparate Impact for Each Attribute')
    ax.set_xticks(x + width * (len(data) - 1) / 2)  # Centering the ticks between groups
    ax.set_xticklabels(data.keys())
    ax.legend(title='Attributes')

    plt.show()

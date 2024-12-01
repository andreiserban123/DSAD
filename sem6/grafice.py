import matplotlib.pyplot as plt
import numpy as np


def plot_varianta(alpha, criterii):
    fig = plt.figure("Plot varianta", figsize=(10, 7))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot varianta", fontdict={"fontsize": 16, "color": "b"})
    m = len(alpha)
    x = np.arange(1, m + 1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r")
    ax.axvline(criterii[0],c="m",label="Acoperire minimala")
    print(np.isnan(criterii[1]))
    if not np.isnan(criterii[1]):
        ax.axvline(criterii[1], c="c", label="Kaiser")
    if criterii[2] is not None:
        ax.axvline(criterii[2], c="k", label="Cattell")
    ax.legend()
    plt.show()

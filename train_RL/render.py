import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def render(data_batch, actions_batch, R_batch, idx_arr, name_suffix=''):
    num_cols = len(idx_arr)

    fig, ax = plt.subplots(nrows=1, ncols=num_cols, figsize=(5 * num_cols, 4))

    # data plotting
    data = [data_batch[i].cpu() for i in idx_arr]
    for i, idx in enumerate(idx_arr):
        x = torch.index_select(data[i], 0, torch.tensor([0]))
        y = torch.index_select(data[i], 0, torch.tensor([1]))
        ax[i].plot(x, y, 'w*')
    
    # roads plotting
    actions = [torch.index_select(a.cpu(), 0, torch.tensor(idx_arr)) for a in actions_batch]
    R = torch.index_select(R_batch.cpu(), 0, torch.tensor(idx_arr))
    for i, idx in enumerate(idx_arr):
        x = [torch.index_select(a[i], 0, torch.tensor([0])) for a in actions]
        y = [torch.index_select(a[i], 0, torch.tensor([1])) for a in actions]
        ax[i].plot(x[0], y[0], 'r^', ms=10) # start
        ax[i].plot(x[-1], y[-1], 'bs', ms=10) # finish
        ax[i].plot(0, 0, 'wo', ms=10) # finish
        ax[i].plot([0] + x + [0], [0] + y + [0])
    
        ax[i].set_title(f'idx = {idx} reward={round(R[i].item(), 2)}')

    # styles
    plt.suptitle('map of the city ' + name_suffix)
    for i, idx in enumerate(idx_arr):
        ax[i].axes.xaxis.set_visible(False)
        ax[i].axes.yaxis.set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_facecolor((0, 0, 0, 1))

    plt.show()
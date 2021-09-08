import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


def random_batch_q(trainloader, batchz):
    ep = 0
    while True:
        for batchx in trainloader:
            batchxz = torch.cat((batchx, batchz), 1)
            yield batchx, batchxz, ep
        ep += 1


def get_train_set(batch_size, trainset):
    zset = np.random.uniform(0.0, 1.0, (batch_size, 1))
    batchz = torch.from_numpy(zset)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    return trainloader, batchz


def random_batch_ab(batch_size, trainset):
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    while True:
        for batch in trainloader:
            yield batch


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def split_data_tv(data,
                  ratio=0.7):
    xy = np.random.permutation(data.shape[0])
    le = np.int_(ratio * data.shape[0])
    idt = xy[:le]
    idv = xy[le:]
    data_t = data[idt]
    data_v = data[idv]
    return data_t, data_v


def split_idx_tv(size,
                 ratio=0.7):
    xy = np.random.permutation(size)
    le = np.int_(ratio * size)
    idt = xy[:le]
    idv = xy[le:]
    return idt, idv


def split_data_tvt(data,
                   ratio1=0.6, ratio2=0.8):
    xy = np.random.permutation(data.shape[0])
    le1 = np.int_(ratio1 * data.shape[0])
    le2 = np.int_(ratio2 * data.shape[0])
    idt = xy[:le1]
    idv = xy[le1:le2]
    idtest = xy[le2:]
    data_t = data[idt]
    data_v = data[idv]
    data_test = data[idtest]
    return data_t, data_v, data_test


def split_idx_tvt(size,
                  ratio1=0.6, ratio2=0.8):
    xy = np.random.permutation(size)
    le1 = np.int_(ratio1 * size)
    le2 = np.int_(ratio2 * size)
    idt = xy[:le1]
    idv = xy[le1:le2]
    idtest = xy[le2:]
    return idt, idv, idtest


def plot_committor_hist(q_values,
                        num_col=10, save_file=-1, fig_size=(10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(q_values, num_col, histtype='bar', ec='black')
    ax.set_xlim([0, 1])
    ax.set_xlabel('$p_{C_{ax}}$', fontsize=18)
    ax.set_ylabel('$P(p_{C_{ax}})$', fontsize=18)
    if save_file != -1:
        fig.savefig(save_file)
    plt.show()


def plot_function(get_fn,
                  xl=-1, xr=1, dx=0.01, save_file=-1, fig_size=(10, 10)):
    xx = np.arange(xl, xr, dx)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(xx, get_fn(xx))
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$f(x)$', fontsize=18)
    if save_file != -1:
        fig.savefig(save_file)
    plt.show()

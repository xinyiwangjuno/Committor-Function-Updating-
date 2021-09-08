import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
# This file is used to simulate data and get V(x).


class Mueller_System(object):
    A = np.array([-0.558223634633, 1.4417258418], dtype=np.float64)
    B = np.array([0.623499404931, 0.0280377585287], dtype=np.float64)
    r = 0.1
    R = 0.12

    aa = [-1, -1, -6.5, -0.7]
    bb = [0, 0, 11, -0.6]
    cc = [-10, -10, -6.5, -0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    sigma = 0.05


    def __init__(self, dim):
        self.dim = dim

    def get_V(self, px):
        px = np.array(px)
        ee = 0
        if np.size(px.shape) == 2:
            for j in range(4):
                ee = ee + self.AA[j] * np.exp(self.aa[j] * (px[:, 0] - self.XX[j]) ** 2 +
                                              self.bb[j] * (px[:, 0] - self.XX[j]) * (px[:, 1] - self.YY[j]) +
                                              self.cc[j] * (px[:, 1] - self.YY[j]) ** 2)
            # ee += 9 * np.sin(2 * 5 * np.pi * px[:, 0]) * np.sin(2 * 5 * np.pi * px[:, 1])
            for i in range(2, self.dim):
                ee += px[:, i] ** 2 / 2 / self.sigma ** 2
        else:
            for j in range(4):
                ee = ee + self.AA[j] * np.exp(self.aa[j] * (px[0] - self.XX[j]) ** 2 +
                                              self.bb[j] * (px[0] - self.XX[j]) * (px[1] - self.YY[j]) +
                                              self.cc[j] * (px[1] - self.YY[j]) ** 2)
            ee += 9 * np.sin(2 * 5 * np.pi * px[0]) * np.sin(2 * 5 * np.pi * px[1])
            for i in range(2, self.dim):
                ee += px[i] ** 2 / 2 / self.sigma ** 2
        return ee

    def get_grad(self, px):
        px = np.array(px)
        gg = np.zeros(shape=(self.dim,), dtype=np.float64)
        for j in range(4):
            ee = self.AA[j] * np.exp(self.aa[j] * (px[0] - self.XX[j]) ** 2 +
                                     self.bb[j] * (px[0] - self.XX[j]) * (px[1] - self.YY[j]) +
                                     self.cc[j] * (px[1] - self.YY[j]) ** 2)
            gg[0] = gg[0] + (2 * self.aa[j] * (px[0] - self.XX[j]) +
                             self.bb[j] * (px[1] - self.YY[j])) * ee
            gg[1] = gg[1] + (self.bb[j] * (px[0] - self.XX[j]) +
                             2 * self.cc[j] * (px[1] - self.YY[j])) * ee
        # gg[0] += 9 * 2 * 5 * np.pi * np.cos(2 * 5 * np.pi * px[0]) * np.sin(2 * 5 * np.pi * px[1])
        # gg[1] += 9 * 2 * 5 * np.pi * np.sin(2 * 5 * np.pi * px[0]) * np.cos(2 * 5 * np.pi * px[1])
        for i in range(2, self.dim):
            gg[i] = px[i] / self.sigma ** 2
        return gg

    def sim_Langevin(self, kBT, dt, D_size, eps, firstsave=1000, t_sep=100):
        Dx = []
        Dy = []
        px = np.zeros((self.dim,))
        py = np.zeros((self.dim,))
        px[:2] = (self.A + self.B)/2
        id_ = 0
        i = 0
        while True:
            px = px - dt * self.get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(self.dim,))

            if i >= firstsave and i % t_sep == 0 and px[0] <= 1 and px[0] >= -1.5 \
                    and px[1] <= 2 and px[1] >= -0.5 and LA.norm(px[:2] - self.A) > self.r + eps and LA.norm(
                px[:2] - self.B) > self.r+eps:
                Dx.append(px)
                py = px - dt * self.get_grad(px) + np.sqrt(dt) * np.random.normal(size=(self.dim,))
                Dy.append(py)
                id_ += 1
                if id_ >= D_size:
                    break
            i += 1

        datax = np.zeros((D_size, 1 + self.dim))
        datay = np.zeros((D_size, 1 + self.dim))
        datax[:, 1:] = np.array(Dx)
        datax[:, 0] = self.get_V(Dx)
        datay[:, 1:] = np.array(Dy)
        datay[:, 0] = self.get_V(Dy)
        # print(np.shape(datax))
        print('generating data finished!')
        return datax, datay

    def sim_adatabc(self, kBT, dt, D_size, eps, firstsave=1000, t_sep=100):
        Da = []
        px = np.zeros((self.dim,))
        px[:2] = self.A
        id_ = 0
        i = 0
        while True:
            px = px - dt * self.get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(self.dim,))

            if i >= firstsave and i % t_sep == 0 and np.abs(LA.norm(px[:2] - self.A) - self.r) < eps:
                Da.append(px)
                id_ += 1
                if id_ >= D_size:
                    break
            i += 1

        adata = np.array(Da)
        print('generating adatabc finished!')
        return adata

    def sim_bdatabc(self, kBT, dt, D_size, eps, firstsave=1000, t_sep=100):
        Db = []
        px = np.zeros((self.dim,))
        px[:2] = self.B
        id_ = 0
        i = 0
        while True:
            px = px - dt * self.get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(self.dim,))

            if i >= firstsave and i % t_sep == 0 and np.abs(LA.norm(px[:2] - self.B) - self.r) < eps:
                Db.append(px)
                id_ += 1
                if id_ >= D_size:
                    break
            i += 1

        bdata = np.array(Db)
        print('generating bdatabc finished!')
        return bdata

    def sim_adatain(self, kBT, dt, D_size, firstsave=1000, t_sep=100):
        Da = []
        px = np.zeros((self.dim,))
        px[:2] = self.A
        id_ = 0
        i = 0
        while True:
            px = px - dt * self.get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(self.dim,))

            if i >= firstsave and i % t_sep == 0 and LA.norm(px[:2] - self.A) < self.r:
                Da.append(px)
                id_ += 1
                if id_ >= D_size:
                    break
            i += 1

        adata = np.array(Da)
        print('generating adatain finished!')
        return adata

    def sim_bdatain(self, kBT, dt, D_size, firstsave=1000, t_sep=100):
        Db = []
        px = np.zeros((self.dim,))
        px[:2] = self.B
        id_ = 0
        i = 0
        while True:
            px = px - dt * self.get_grad(px) + np.sqrt(2 * kBT * dt) * np.random.normal(size=(self.dim,))

            if i >= firstsave and i % t_sep == 0 and LA.norm(px[:2] - self.B) < self.r:
                Db.append(px)
                id_ += 1
                if id_ >= D_size:
                    break
            i += 1

        bdata = np.array(Db)
        print('generating bdatain finished!')
        return bdata


    def show_V_q(self, kBT, fig_size=(8, 5), aspect=0.8, save_file=-1, levels_V=50, levels_q=10, add_text=-1):
        D_test = []
        xx = np.arange(-1.5, 1.0, 0.01)
        yy = np.arange(-0.5, 2.0, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        sizes = np.shape(XX)
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                D_test.append([xx[j], yy[i]] + [0] * (self.dim - 2))
        ZZ = np.reshape(self.get_V(np.array(D_test)), sizes)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.contour(XX, YY, ZZ, levels_V)
        true_q = np.loadtxt('TrueSol/' + str(kBT) + '.txt').reshape(sizes)
        CS2 = ax.contourf(XX, YY, true_q, levels_q, cmap='Greens')
        plt.colorbar(CS2)
        ax.add_artist(plt.Circle(tuple(self.A), self.r, color='k'))
        ax.add_artist(plt.Circle(tuple(self.B), self.r, color='k'))
        ax.text(self.A[0] - 0.1, self.A[1] + 0.15, '$A$', fontsize=28)
        ax.text(self.B[0] + 0.1, self.B[1] - 0.20, '$B$', fontsize=28)
        ax.set_xlabel('$x_{1}$', fontsize=18)
        ax.set_ylabel('$x_{2}$', fontsize=18, rotation=0)
        ax.set_xlim([-1.5, 1])
        ax.set_ylim([-0.5, 2])
        if add_text != -1:
            ax.text(0.08, 0.92, add_text, ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.set_aspect(aspect=aspect)
        plt.tight_layout()
        fig.subplots_adjust(left=.178)
        if save_file != -1:
            fig.savefig(save_file, format='png', dpi=100)
        plt.show()

    def show_data(self, data, aspect=0.8, save_file=-1, fig_size=(10, 10)):
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(data[:, 1], data[:, 2], c='g', alpha=0.5, s=1)
        ax.add_artist(plt.Circle(tuple(self.A), self.r, color='k'))
        ax.add_artist(plt.Circle(tuple(self.B), self.r, color='k'))
        ax.text(self.A[0] - 0.1, self.A[1] + 0.15, '$A$', fontsize=28)
        ax.text(self.B[0] + 0.1, self.B[1] - 0.20, '$B$', fontsize=28)
        ax.set_xlabel('$x_{1}$', fontsize=18)
        ax.set_ylabel('$x_{2}$', fontsize=18, rotation=0)
        ax.set_xlim([-1.5, 1])
        ax.set_ylim([-0.5, 2])
        ax.set_aspect(aspect=aspect)
        plt.tight_layout()
        if save_file != -1:
            fig.savefig(save_file, format='png', dpi=100)
        plt.show()

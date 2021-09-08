from torch.optim import Adam
from torch.autograd import grad
from models import *
from myutils import *
from Muller import Mueller_System
from torch.autograd import Variable
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


class Muller_Solver():
    def __init__(
            self, q, disc, data_x_np, data_y_np, data_a, data_b, batch_size, dim, n_disc, disc_lr,
            gen_lr, beta1, beta2, lam, lam_bc
    ):
        self.dim = dim
        trainloader_x, batchz = get_train_set(batch_size, data_x_np)
        self.data_xz = random_batch_q(trainloader_x, batchz)
        trainloader_y, _ = get_train_set(batch_size, data_y_np)
        self.data_yz = random_batch_q(trainloader_y, batchz)
        self.data_z_tensor = batchz
        self.batch_size = batch_size
        self.data_a = random_batch_ab(np.shape(data_a)[0], data_a)
        self.data_b = random_batch_ab(np.shape(data_b)[0], data_b)
        self.q = q
        self.disc = disc
        self.n_disc = n_disc
        self.q_opt = Adam(
            self.q.parameters(), lr=gen_lr, betas=(beta1, beta2)
        )
        self.disc_opt = Adam(
            self.disc.parameters(), lr=disc_lr, betas=(beta1, beta2)
        )
        self.lam = lam
        self.lam_bc = lam_bc
        self.ep = 0
        self.q_losses = []
        self.disc_losses = []
        self.grad_pens = []
        self.mus = Mueller_System(self.dim)

    def saving_pars(self, NN_file_q, NN_file_disc):
        open(NN_file_q, 'w')
        torch.save(self.q, NN_file_q)
        open(NN_file_disc, 'w')
        torch.save(self.disc, NN_file_disc)

    def train(self, num_epochs, NN_file_q, NN_file_disc):
        while self.ep < num_epochs:
            self._train_epoch()
        self._plot()
        self.saving_pars(NN_file_q, NN_file_disc)

    def _train_epoch(self):
        curr_ep = self.ep
        step_num = 0
        while self.ep == curr_ep:
            self._train_step(step_num)
            step_num += 1

    def _train_step(self, step_num):
        disc_loss, grad_pen = self._disc_step()
        q_loss = self._q_step()
        self._record(disc_loss, q_loss, grad_pen)

    def _disc_step(self):
        self.q.train()
        self.disc.train()

        freeze_params(self.q)
        unfreeze_params(self.disc)

        for __ in range(self.n_disc):
            xsample, xzsample, self.ep = next(self.data_xz)
            ysample, yzsample, _ = next(self.data_yz)

            disc_outputs = self.disc(xzsample).reshape(-1, 1)

            q_x = self.q(xsample)
            q_y = self.q(ysample)

            grad_pen = self._grad_pen(xzsample, yzsample)
            print(disc_outputs.size())
            disc_loss = ((q_x - q_y).reshape(-1, 1)) * (disc_outputs)
            print(disc_loss.size())
            disc_loss = (2 * self.data_z_tensor - 1) * disc_loss
            disc_loss = -2 * disc_loss + grad_pen
            disc_loss = disc_loss.mean()

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()

            return disc_loss, grad_pen

    def _grad_pen(self, xsample, ysample):
        epsilon = np.random.uniform(
            size=(xsample.shape[0], 1))
        epsilon = torch.from_numpy(epsilon).float()
        xhat = epsilon * xsample + (1 - epsilon) * ysample
        xhat.requires_grad_()

        lipschitz_grad = grad(
            outputs=self.disc(xhat).sum(),
            inputs=xhat,
            create_graph=True,
            retain_graph=True)[0]
        lipschitz_grad = lipschitz_grad.view(xhat.shape[0], -1)
        grad_norm = torch.sqrt(torch.sum(lipschitz_grad ** 2, dim=1) + 1e-12)
        grad_pen = self.lam * (grad_norm - 1) ** 2
        return grad_pen.reshape(-1, 1)

    def get_q(self, px, q):
        px = torch.from_numpy(px)
        if np.size(np.shape(px)) == 1:
            # print(type(q(px)))
            return q(px).detach().numpy()[0]
        else:
            return q(px).detach().numpy()

    def get_g(self, px, q):
        px = torch.from_numpy(px)
        px = Variable(px, requires_grad=True)
        g = grad(
            outputs=q(px),
            inputs=px,
            create_graph=True,
            retain_graph=True)[0]
        if np.size(np.shape(px)) == 1:
            # print(g)
            return g.detach().numpy()[0]
        else:
            return g.detach().numpy()

    def _q_step(self):
        self.q.train()
        self.disc.train()

        unfreeze_params(self.q)
        freeze_params(self.disc)

        self.q_opt.zero_grad()

        xsample, xzsample, _ = next(self.data_xz)
        ysample, _, _ = next(self.data_yz)
        adata = next(self.data_a)
        bdata = next(self.data_b)

        disc_outputs = self.disc(xzsample).reshape(-1, 1)
        bc_a = torch.abs(self.q(adata))
        bc_b = torch.abs(1 - self.q(bdata))
        bc_pen = bc_a.mean() + bc_b.mean()

        q_x = self.q(xsample)
        q_y = self.q(ysample)

        gen_loss = (2 * self.data_z_tensor - 1) * disc_outputs
        print(gen_loss.size())
        gen_loss = gen_loss * ((q_x - q_y).reshape(-1, 1))
        print(gen_loss.size())
        gen_loss = 2 * gen_loss.mean() + self.lam_bc * bc_pen
        gen_loss.backward()

        self.q_opt.step()
        return gen_loss

    def _record(self, disc_loss, q_loss, grad_pen):
        # record statistics
        self.q_losses.append(q_loss.item())
        self.disc_losses.append(disc_loss.item())
        self.grad_pens.append(grad_pen.mean().item())

        # if step_num % self.print_freq == 0:
        print('discriminator loss: ', disc_loss.item())
        print('grad_pen', grad_pen.mean().item())
        print('generator loss: ', q_loss.item())
        print('------------------------')

    def _plot(self):
        # plot losses
        plt.figure()
        plt.plot(self.q_losses, label='generator loss')
        plt.xlabel('batch number')
        plt.ylabel('mean loss for batch')
        plt.legend()
        plt.savefig('./plots/losses_generator.png')
        plt.figure()
        plt.plot(self.disc_losses, label='discriminator loss')
        plt.plot(self.grad_pens, label='gradient penalty')
        plt.xlabel('batch number')
        plt.ylabel('mean loss for batch')
        plt.legend()
        plt.savefig('./plots/losses_discriminator.png')

    def show_contour_lines(self, NN_file_q, kBT, levels_q=[0.5], aspect=0.8, save_file=-1,
                           levels_V=50,
                           fig_size=(8, 5), add_text=-1):
        grids = []
        q = torch.load(NN_file_q)
        xx = np.arange(-1.5, 1.0, 0.01)
        yy = np.arange(-0.5, 2.0, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        sizes = np.shape(XX)
        print(sizes)
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                grids.append([xx[j], yy[i]] + [0] * (self.dim - 2))
        ZZ = np.reshape(self.mus.get_V(np.array(grids)), sizes)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.contour(XX, YY, ZZ, levels_V)
        myq = (q(torch.from_numpy(np.array(grids)))).detach().numpy()
        np.savetxt('output/Q_results', myq.reshape(sizes), fmt='%s')
        # 这一部分备注可用于画出真解，只要将真解放在TrueSol文件夹即可。
        ax.contour(XX, YY, np.loadtxt('TrueSol/10.txt')[:-1, :-1],
                   levels_q, colors='red', linewidths=3)
        ax.contour(XX, YY, q(torch.from_numpy(np.array(grids))).detach().numpy().reshape(sizes),
                   levels_q, colors='blue', linewidths=3, linestyles='dashed')
        ax.add_artist(plt.Circle(self.mus.B, self.mus.r, color='k'))
        ax.text(self.mus.A[0] - 0.1, self.mus.A[1] + 0.15, '$A$', fontsize=28)
        ax.text(self.mus.B[0] + 0.1, self.mus.B[1] - 0.20, '$B$', fontsize=28)
        if add_text != -1:
            ax.text(0.08, 0.92, add_text, ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.set_xlabel('$x_{1}$', fontsize=18)
        ax.set_ylabel('$x_{2}$', fontsize=18, rotation=0)
        ax.set_xlim([-1.5, 1])
        ax.set_ylim([-0.5, 2])
        ax.set_aspect(aspect=aspect)
        plt.tight_layout()
        if save_file != -1:
            fig.savefig(save_file, format='png', dpi=100)
        plt.show()

    def plot_qsurface(self, NN_file_q):
        grids = []
        q = torch.load(NN_file_q)
        xx = np.arange(-1.5, 1.0, 0.01)
        yy = np.arange(-0.5, 2.0, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        sizes = np.shape(XX)
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                grids.append([xx[j], yy[i]] + [0] * (self.dim - 2))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        QQ = q(torch.from_numpy(np.array(grids))).detach().numpy()
        surf = ax.plot_surface(XX, YY, QQ.reshape(sizes), linewidth=0, antialiased=False)
        ax.set_zlim(0.0, 1.0)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        QQ2 = np.loadtxt('TrueSol/10.txt')[:-1, :-1]
        surf2 = ax2.plot_surface(XX, YY, QQ2.reshape(sizes), linewidth=0, antialiased=False)
        ax2.set_zlim(0.0, 1.0)

        ax2.zaxis.set_major_locator(LinearLocator(10))
        ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig2.colorbar(surf2, shrink=0.5, aspect=5)

        plt.show()

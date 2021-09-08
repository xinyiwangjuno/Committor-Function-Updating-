import os
import Muller
import numpy as np
import Train
from models import CommittorNet, Discriminator
import torch

ntest = 5
num_epochs = 100
batch_size = 200
disc_lr = 1e-6
gen_lr = 1e-6
beta1 = 0.5
beta2 = 0.9
n_disc = 10
lam = 0.01
lam_bc = 500
dim = 2
D_size = 1000
meta_num = 2000
os.makedirs('./plots', exist_ok=True)

'''
    Data_a are data from the bdy of set A, and data_b are those from bdy of B.
    These two datasets are generated to satisfy the bdy conditons.
'''

data_a = np.loadtxt('output/data_a_bc' + '_' + str(D_size)+'_'+str(dim)).reshape((-1,dim))
data_b = np.loadtxt('output/data_b_bc' + '_' + str(D_size)+'_'+str(dim)).reshape((-1,dim))




test = 0
q = CommittorNet(dim, 50, thresh=torch.sigmoid)
disc = Discriminator(dim+1, 50)

M_sys = Muller.Mueller_System(dim)
data_x, data_y = M_sys.sim_Langevin(kBT=10, dt=1e-6, D_size=D_size, eps = 0.01)

data_x_train = np.concatenate((data_x[:,1:], data_a, data_b), 0)
data_y_train = np.concatenate((data_y[:,1:], data_a, data_b), 0)


np.savetxt('output/data_x_' + str(test) + '_' + str(D_size), data_x)
np.savetxt('output/data_y_' + str(test) + '_' + str(D_size), data_y)

NN_file_q = 'output/saved_models_q/' + str(test) + '_' + str(D_size)
NN_file_disc = 'output/saved_models_disc/' + str(test) + '_' + str(D_size)
M_sol = Train.Muller_Solver(q, disc, data_x_train, data_y_train, data_a, data_b, batch_size, dim, n_disc, disc_lr, gen_lr, beta1,
                                beta2, lam, lam_bc)
M_sol.train(num_epochs, NN_file_q, NN_file_disc)

del M_sys, data_x, data_y



# Plottings

NN_file_q = 'output/saved_models_q/' + str(test) + '_' + str(int(1000))
M_sol.show_contour_lines(NN_file_q=NN_file_q, kBT=10, fig_size=(8, 5),
                               save_file='Mueller_highT.png', add_text='(c)')
M_sol.plot_qsurface(NN_file_q)


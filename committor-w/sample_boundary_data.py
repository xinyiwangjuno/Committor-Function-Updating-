import Muller
import numpy as np

ntest = 10
num_epochs = 100
batch_size = 100
disc_lr = 1e-5
gen_lr = 1e-5
beta1 = 0.3
beta2 = 0.6
n_disc = 1
lam = 0.01
lam_bc = 500
dim = 2
D_size = 1000
meta_num = 2000

test = 0


'''

Sample data within set A and B and data on the boundary of set A and B

'''


M_sys = Muller.Mueller_System(dim)
data_a_bc = M_sys.sim_adatabc(kBT=10, dt=1e-6, D_size = 2*D_size, eps = 0.02)
data_b_bc = M_sys.sim_bdatabc(kBT=10, dt=1e-6, D_size = 2*D_size, eps = 0.02)
data_a_in = M_sys.sim_adatain(kBT=10, dt=1e-6, D_size = 2*D_size)
data_b_in = M_sys.sim_bdatain(kBT=10, dt=1e-6, D_size = 2*D_size)

np.savetxt('output/data_a_bc_sin' + str(test) + '_' + str(D_size)+'_'+str(dim), data_a_bc)
np.savetxt('output/data_b_bc_sin' + str(test) + '_' + str(D_size)+'_'+str(dim), data_b_bc)
np.savetxt('output/data_a_in_sin' + str(test) + '_' + str(D_size)+'_'+str(dim), data_a_in)
np.savetxt('output/data_b_in_sin' + str(test) + '_' + str(D_size)+'_'+str(dim), data_b_in)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['export OPENBLAS_NUM_THREADS']='1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch import Tensor
from typing import List, Tuple
from models.models import MWT2d
from models.utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import cv2
import glob
from functools import partial
import matplotlib as ml
from PIL import Image
from models.utils_3d import train, test, LpLoss, get_filter, UnitGaussianNormalizer
import operator
from functools import reduce
from timeit import default_timer
import matplotlib
matplotlib.use('agg')
import pickle
import tqdm
import psutil

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

r = 1
h = int(((64 - 1)/r) + 1)
s = h

dataloader = np.load('Data/melt/melt_rate.npy')
u_data = dataloader.astype(np.float32)
x_train = torch.from_numpy(u_data[:170, ::r,::r, 0])
y_train = torch.from_numpy(u_data[:170, ::r,::r, 1])
x_test = torch.from_numpy(u_data[-30:, ::r,::r, 0])
y_test = torch.from_numpy(u_data[-30:, ::r,::r, 1])
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

ich = 3
initializer = get_initializer('xavier_normal') # xavier_normal, kaiming_normal, kaiming_uniform

torch.manual_seed(0)
np.random.seed(0)

model = MWT2d(ich, 
            alpha = 12,
            c = 4,
            k = 4, 
            base = 'legendre', # 'chebyshev'
            nCZ = 4,
            L = 0,
            initializer = initializer,
            ).to(device)
learning_rate = 0.001
epochs = 2000
step_size = 100
gamma = 0.5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False)



def fourier(acc_rates_mean):

    data = np.ones((64, 64)) * acc_rates_mean
    data = np.reshape(data, (1, data.shape[0], data.shape[1]))
    x_test = data.astype(np.float32)
    x_test = torch.from_numpy(x_test)
    
    x_test = x_normalizer.encode(x_test)
    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(1,s,s,1), grid.repeat(1,1,1,1)], dim=3)
    x_test = torch.tensor(x_test, dtype=torch.float)
    
    checkpoint = torch.load('NS_models/melt/melt.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    y_normalizer.cuda()

    pred = torch.zeros(x_test[:, :, :, 0].shape)
    post_proc=y_normalizer.decode
    index = 0
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            out = model(x)
            output = post_proc(out)
            pred[index] = output
            return pred[index].numpy()
        

def rel(x, y):
    l = x.shape[0]*x.shape[1]
    # L2 norm
    diff_norms = np.linalg.norm(x.reshape((1,l)) - y.reshape((1,l)))
    return diff_norms

def log_likelihood(y, f, sigma):
    l2_total = math.pow(rel(y[:,:,1], f),2)
    
    scale = 1 #0.000001
    return -0.5*l2_total/(math.pow(sigma,2))*scale

def pcn(N, u0, y, n_iters, beta, sigma):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    # print(f"Initialisation:{u_prev}")

    # replace with operator
    f = fourier(u_prev)

    ll_prev = log_likelihood(y, f, sigma)
    # print(f"log-likelihood initial is: {ll_prev}")
    print(f"Initial proposal: {u_prev}")
    count = 0
    for i in tqdm.trange(n_iters):
        u_new = np.sqrt(1 - pow(beta, 2)) * u_prev + beta * np.random.normal(0, scale = 0.01) # Propose new sample using pCN proposal
        
        f = fourier(u_new)

        ll_new = log_likelihood(y, f, sigma)

        # Calculate pCN acceptance probability
        log_alpha = min(0, ll_new-ll_prev)
        print('ll_new: ', ll_new)
        print('ll_prev: ', ll_prev)
        print('log_alpha: ', log_alpha)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_u<=log_alpha # Compare log_alpha and log_u to accept/reject sample (accept should be boolean)

        if accept:
            acc += 1
            X.append(u_new)
#             print("Previously: ", u_prev)
            u_prev = u_new
            ll_prev = ll_new
            beta = min(1, 1.1*beta)
#             print("Proposal: ", u_new)
#             print("Step {} accepted".format(i+1))
            count = count-1
#             if count > 50:
#                 print("Proposal: ", u_new)
#                 print(f"log-likelihood is: {ll_new}")
#                 beta = 0.2
#             else:
#                 pass
            
        else:
            X.append(u_prev)
            beta = max(0.075, 0.75*beta)
            count = count+1
#             if count > 50:
#                 print("Proposal: ", u_new)
#                 print(f"log-likelihood is: {ll_new}")
#                 beta = 0.2
#             else:
#                 pass
# #             print("Step {} rejected".format(i+1))
            
#         if count > 200:
#             print("Proposal: ", u_new)
#             print(f"log-likelihood is: {ll_new}")
#             break
#         else:
#             pass

    return X, acc / n_iters

# # # Load noisy data
y = np.load('Data/melt/melt_rate.npy')
# y = np.load("firefigs/inverse/fno_training/data/gaussian/noisy_1d.npy")


# Set Gaussian prior around true accumulation rates
# Covariance
# Simulate with 2 although true rate is 1.5
true_acc_rate = 10
N = 64

# Set number of iterations and step size beta
n_iters = 10000
beta = 0.12

# Likelihood variance
sigma = 1

# Run MCMC
simulations = []
for k in range(3, y.shape[0]):
    observed_data = y[k, : , :]
    u0 = np.random.normal(true_acc_rate, 1)
    pcn_u, pcn_acc = pcn(N, u0, observed_data, n_iters, beta, sigma)
    pcn_u = np.array(pcn_u)
    simulations.append(pcn_u)
    np.save('simulations/melt_new/sim{}.npy'.format(k), pcn_u)
    print("Step: ", k)
    print("Acceptance rate: ", pcn_acc)
simulations = np.array(simulations)
np.save('simulations/melt_new/sim.npy', simulations)
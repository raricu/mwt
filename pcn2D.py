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
np.random.seed(1000)

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

dataloader = np.load('Data/noisy_melt.npy')
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

def create_matrix(melt, temp):
    a = np.empty((64, 64))
    rows= np.linspace(melt*100, temp, a.shape[1])
    for i in range(a.shape[0]):
        a[i, :] = np.ones(a.shape[1])*rows[i]
    return a

def fourier(u):
    
    melt = u[0]
    temp = u[1]
    data = create_matrix(u[0], u[1])
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
    
    checkpoint = torch.load('NS_models/final_exp/altered_model1700.pt')
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
    # print(x.shape)
    # print(y.shape)
#     l = x.shape[0]*x.shape[1]
    # L2 norm
#     diff_norms = np.linalg.norm(x.reshape((1,l)) - y.reshape((1,l)))
    diff = x - y
    return np.linalg.norm(diff, ord = 2)

def log_likelihood(y, f, sigma):
    l2_total = math.pow(rel(y, f),2)
    
    scale = 1 #0.000001
    return -0.5*l2_total/(math.pow(sigma,2))*scale

def pcn(y, n_iters, beta, sigma):
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
    np.random.seed(14235)
    X = []
    acc = 0
    melt = 5 #1.5
    temp = 273
    mean = (melt, temp)
    cov = [[1, 0], [0, 1]]
    u0 = np.random.multivariate_normal(mean, cov)
    u_prev = u0

    ll_prev = log_likelihood(y, fourier(u_prev), sigma)
    print(f"Initial proposal: {u_prev}")
    u_new = np.ones(u_prev.shape)
    count = 0
    for i in tqdm.trange(n_iters):
#         print("beta: ", beta)
        xi = np.random.multivariate_normal((0, 0), cov)
        u_new = np.sqrt(1-beta**2)*u_prev + beta * xi # Propose new sample using pCN proposal
#         print("xi:", xi)
#         print('u_prev:', u_prev)
#         print('u_new:', u_new)

        ll_new = log_likelihood(y, fourier(u_new), sigma)

        # Calculate pCN acceptance probability
        log_alpha = min(0, ll_new-ll_prev) 
        log_u = np.log(np.random.random())
#         print('ll_prev: ', ll_prev)
#         print('ll_new: ', ll_new)
#         print('log_u < log_alpha: ', log_u < log_alpha)
#         print('log_alpha: ', log_alpha)
#         print('log_u: ' , log_u)
#         print('beta: ', beta)
        accept = log_u<=log_alpha # Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
#             beta = min(1, 1.001*beta)

            count = count-1
#             if count > 50:
#                 print("Proposal: ", u_new)
#                 print(f"log-likelihood is: {ll_new}")
#                 beta = 0.2
#             else:
#                 pass
            
        else:
            X.append(u_prev)
#             beta = max(0.001, 0.5*beta)
            count = count+1
#             if count > 50:
# #                 print("Proposal: ", u_new)
# #                 print(f"log-likelihood is: {ll_new}")
#                 beta = 0.0001
#                 count = 0
#             else:
#                 pass
            
#         if count > 200:
#             print("Proposal: ", u_new)
#             print(f"log-likelihood is: {ll_new}")
#             break
#         else:
#             pass
    return X, acc / n_iters


# # # Load noisy data
y = np.load('Data/2D_altered.npy')


# Run MCMC
n_iters = 1000000
# n_iters = 10
beta = 0.02

# Likelihood variance
sigma = 1

simulations = []
for k in range(y.shape[0]):
    print("Step: ", k)
    observed_data = y[k, : , :, 1]
    print("parameters: ", y[k, 0, 0, 0], y[k, -1 , -1, 0])
    print(observed_data.shape)
    pcn_u, pcn_acc = pcn(observed_data, n_iters, beta, sigma)
    pcn_u = np.array(pcn_u)
    print("Acceptance rate: ", pcn_acc)
    simulations.append(pcn_u)
    np.save('simulations/2D/sim{}.npy'.format(k), pcn_u)
simulations = np.array(simulations)
np.save('simulations/2D/sim.npy', simulations)

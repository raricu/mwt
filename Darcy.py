import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['export OPENBLAS_NUM_THREADS']='2'

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.io import loadmat, savemat
import math
import os
import h5py
import matplotlib.pyplot as plt
from functools import partial
from models.models import MWT2d
from models.utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer

print(torch.__version__)

torch.manual_seed(0)
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

def plot_loss(initial, prediction, test, index):
    initial = initial[index, :, :, 0]
    test = test[index]
    prediction = prediction[index]
    loss = abs(torch.sub(test, prediction))

    cp1 = plt.matshow(initial)
    cp2 = plt.matshow(test)
    cp3 = plt.matshow(prediction)
    cp4 = plt.matshow(loss)
    plt.colorbar(cp1)
    plt.colorbar(cp2)
    plt.colorbar(cp3)
    plt.colorbar(cp4)
    
    
data_path = 'Data/2D_new.npy'
ntrain = 350
ntest = 50

r = 1
h = int(((64 - 1)/r) + 1)
s = h

dataloader = np.load(data_path)
print(dataloader.shape)
u_data = dataloader.astype(np.float32)

x_train = torch.from_numpy(u_data[:ntrain, ::r,::r, 0])
y_train = torch.from_numpy(u_data[:ntrain, ::r,::r, 1])

x_test = torch.from_numpy(u_data[-ntest:, ::r,::r, 0])
y_test = torch.from_numpy(u_data[-ntest:, ::r,::r, 1])

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

batch_size = 10
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)

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
epochs = 5000
step_size = 100
gamma = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

train_loss = []
test_loss = []
for epoch in range(1, epochs+1):
    train_l2 = train(model, train_loader, optimizer, epoch, device,
        lossFn = myloss, lr_schedule = scheduler,
        post_proc = y_normalizer.decode)
    train_loss.append(train_l2)
    test_l2 = test(model, test_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
    print(f'epoch: {epoch}, train l2 = {train_l2}, test l2 = {test_l2}')
    test_loss.append(test_l2)
    if epoch%100 == 0:
        PATH = 'NS_models/2D/NS_model{}.pt'.format(epoch)
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': myloss}, PATH)
        np.save('visual/train_loss_2D.npy', train_loss)
        np.save('visual/test_loss_2D.npy', test_loss)
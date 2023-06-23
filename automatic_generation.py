from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from dolfin import Mesh
from tqdm import tqdm
from natsort import natsorted 
from test import data_generation
from reshape import single_batch
from reshape import reshape

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['export OPENBLAS_NUM_THREADS']='2'


rhos = [5, 10, 25, 50]
mus = [0.025, 0.05, 0.1, 0.5, 1, 5, 10]
for rho in rhos:
    for mu in mus:
        data_generation(rho, mu)
        reshape(rho, mu)
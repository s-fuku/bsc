import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.sparse as sparse
import scipy.special as special

from functools import partial

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import graph_tool.all as gt

from datetime import timedelta

import pickle

import os
import tqdm
import sys
sys.path.append('../../util')

from bsc import calc_paracomp_bern_with_prior_beta
from evaluation import calc_auc_main
from cython_normterm_discrete import create_fun_with_mem

outdir = './output'

if not os.path.exists(outdir):
    os.makedirs(outdir)

def calc_codelength_integer(k):
    codelength = 2.865
    k = np.log(k)
    while k > 0:
        codelength += k
        k = np.log(k)
    
    return codelength

with open('data/count_od_by_week.pkl', 'rb') as f:
    count_od = pickle.load(f)

datetimes_array = sorted(count_od['date'].unique())
entity_array = np.unique(np.hstack((count_od['entity1'].unique(), count_od['entity2'].unique())))

λ = 1.0 
a = 0.5
b = 0.5
k_list = [15, 20, 25, 30, 35]

norm_multinom = create_fun_with_mem()

model = BSC()
model.fit(count_od, k_list, directed=False)
model.stats = [np.nan]
model.calc_statistics(k_list, λ, a, b)


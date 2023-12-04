import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.sparse as sparse
import scipy.special as special

from functools import partial

import graph_tool.all as gt

import networkx as nx

from importlib import reload
import os
import pickle
import joblib
import tqdm

import sys
sys.path.append('../util')

import bsc
import generate_data
reload(bsc)
reload(generate_data)
from bsc import BSC, calc_paracomp_bern_with_prior_beta
from evaluation import calc_auc_main
from generate_data import DataGeneratorSBM
from cython_normterm_discrete import create_fun_with_mem
from calc_codelen import calc_codelength_integer


outdir = 'output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

K = 3
N = 200
a = 0.5
b = 0.5
du = 0.1

T = 30
T1 = 15
T2 = 20

k_list = np.array([2, 3, 4])

λ = 1.0

gen = DataGeneratorSBM(T)

normterm = create_fun_with_mem()

n_trial = 10

norm_multinom = create_fun_with_mem()

normterm_y = np.array([np.log(np.sum([
                                    np.exp(
                                        special.loggamma(k**2+1) - 
                                        special.loggamma(n_plus+1) - 
                                        special.loggamma(k**2-n_plus+1) + 
                                        (n_plus + (a-1)) * np.log((n_plus + (a-1))/(k**2+a+b+λ-2)) +
                                        (k**2-n_plus+b-1+λ) * np.log((b-1+λ + k**2-n_plus)/(k**2+a+b+λ-2))
                                    )
                                  for n_plus in range(int(np.floor(2.0-a)), min(int(np.floor(b-1.0+λ+k**2)), k**2))])) for k in k_list])
normterm_z = np.log(norm_multinom.evaluate(N, 3))

stats_list = []
for n in range(n_trial):
    X_list, Z_list, eta_former, eta_latter = gen.generate(K, N, T1, T2, a, b, du, ratio, seed=n)

    df_list = []
    datetimes = pd.date_range('2023-01-01', freq='1W', periods=len(X_list))
    for dt, X in zip(datetimes, X_list):
        indices_i, indices_j = np.nonzero(X)
        df = pd.DataFrame({'datetime': dt, 'V1': indices_i, 'V2': indices_j})
        df_list.append(df)

    df_all = pd.concat(df_list, axis=0)

    model = BSC()
    model.fit(df_all, k_list, directed=True)
    
    model.calc_statistics(k_list, λ, a, b)

	stats_list.append(model.stats)

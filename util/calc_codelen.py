import numpy as np

def calc_codelength_integer(k):
    codelength = 2.865
    k = np.log(k)
    while k > 0:
        codelength += k
        k = np.log(k)
    
    return codelength
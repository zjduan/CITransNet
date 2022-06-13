import os
import numpy as np
import torch
from scipy.stats import truncnorm
from tqdm import tqdm
from IPython import embed

def gen(n, m, d=10, phase='train', n_data=200000):
    dir = f'../data_multi/{d}d_{n}x{m}/{phase}_{n_data}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    raw_valuation = torch.rand(n_data, n, m)
    tau = -1 + 2 * torch.rand(n_data, n, d)
    omega = -1 + 2 * torch.rand(n_data, m, d)

    valuation = raw_valuation * torch.sigmoid(tau @ omega.permute(0, 2, 1))
    np.save(os.path.join(dir, 'trueValuations'), valuation.numpy())
    np.save(os.path.join(dir, 'Agent_names_idxs'), tau.numpy())
    np.save(os.path.join(dir, 'Object_names_idxs'), omega.numpy())

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    n, m = 2, 5
    gen(n, m, phase='test', n_data=5000)
    gen(n, m, phase='training', n_data=int(1e5))
    n, m = 3, 10
    gen(n, m, phase='test', n_data=5000)
    gen(n, m, phase='training', n_data=int(1e5))
    n, m = 5, 10
    gen(n, m, phase='test', n_data=5000)
    gen(n, m, phase='training', n_data=int(1e5))

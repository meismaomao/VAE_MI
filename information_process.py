from mutual_info_estimation import *
import numpy as np
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from tensorflow.examples.tutorials.mnist import input_data

NUM_CORES = multiprocessing.cpu_count()

def calc_information_for_sampling(data, bins, pxs,wspxs, unique_inverse_x):
    params = []
    bins = bins.astype(np.float)
    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) -1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
    np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(np.sum(unique_counts))
    PXs = np.array(pxs).T
    local_IXT = calc_information_from_mat(PXs, p_ts, digitized, unique_inverse_x, wspxs)
    params.append(local_IXT)
    return params

def extract_probs(x):
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
    np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pxs = unique_counts / float(np.sum(unique_counts))
    return b1, b, unique_a, unique_inverse_x, pxs

def get_information(ws, x, num_of_bins=10, calc_parallel=True):
    print("start calculating the infromation!")
    bins = np.linspace(0, 1, num_of_bins)
    b1, b, unique_a, unique_inverse_x, pxs = extract_probs(x)
    digitized = bins[np.digitize(np.squeeze(ws.reshape(1, -1)), bins) - 1].reshape(len(ws), -1)
    wsb1, wsb, wsunique_a, wsunique_inverse_x, wspxs = extract_probs(digitized)
    
    if calc_parallel:
        params = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_information_for_sampling)(
            data=ws[i], bins=bins, pxs=pxs, wspxs=wspxs,  unique_inverse_x=unique_inverse_x) for i in range(len(ws))))
    else:
        params = np.array(calc_information_for_sampling(
            data=ws, bins=bins, pxs=pxs,  unique_inverse_x=unique_inverse_x))
    return params

# def main():
#
#     mnist = input_data.read_data_sets('/data/mnist_data/', one_hot=True)
#     data = np.concatenate((mnist.train.images, mnist.test.images), axis=0)[32:128, :]
#     # bins = np.linspace(0, 1, 10)
#     ws = np.loadtxt(r"D:\YQL\DeepLearning\TensorflowTestLearning\VAE_MI\dict_save00019\dw1.txt")
#     # ws = np.array(ws)
#     mi = get_information(ws, data)
#     mi = np.reshape(mi, [1,-1])
#     print(mi)
#
# if __name__ == '__main__':
#     main()


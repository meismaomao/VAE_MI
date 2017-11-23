import numpy as np
import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()

def calc_entropy_for_specipic_t(current_ts, px_i):
    'calc the entropy for special T(hidden layer output value)'
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
    np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.array(p_current_ts, dtype=np.float).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X

def calc_condition_entropy(px, t_data, unique_inverse_x):
    H2X_array = np.array(Parallel(n_jobs=NUM_CORES)(
        delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :],px[i])
        for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X

def calc_information_from_mat(px, ps2, data, unique_inverse_x, wspxs):
    H_data = -np.sum(wspxs * np.log2(wspxs))
    # print(H_data)
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condition_entropy(px, data, unique_inverse_x)
    IX = (H2 - H2X) / np.sqrt(H_data * H2)
    return IX

import multiprocessing as mp
from itertools import repeat
import numpy as np
import random


def multiprocess_data_and_labels(func, data_list, label_list, *args):
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs - 1)
    result = pool.starmap(func, zip(data_list, label_list, *[repeat(x) for x in args]))
    pool.close()
    pool.join()
    return result

def multiprocess_data(func, data_list, *args):
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs - 1)
    result = pool.starmap(func, zip(data_list, *[repeat(x) for x in args]))
    pool.close()
    pool.join()
    return result


def split_val_from_train(data, labels, seed, ratio=0.20):
    z = list(zip(data, labels))
    n_samples = len(z)

    random.seed(seed)
    random.shuffle(z)

    data_s, labels_s = zip(*z)

    indx = int(n_samples * ratio)

    val_data = data_s[:indx]
    val_labels = labels_s[:indx]

    train_data = data_s[indx:]
    train_labels = labels_s[indx:]

    return train_data, train_labels, val_data, val_labels


def clip_ct_window(np_arr, ct_min, ct_max):
    np_arr = np.minimum(255, np.maximum(0, (np_arr - ct_min) / (ct_max - ct_min) * 255))
    return np_arr
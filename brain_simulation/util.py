import numpy as np
import scipy
import jax
import jax.numpy as jnp
from absl import logging 
from bspline import B_spline_bases
from penalty import smoothness_penalty  
import matplotlib.pyplot as plt
from jax.scipy.special import gammaln
import jax.random as random
from jax.scipy.optimize import minimize
from jaxopt import ScipyMinimize
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from pympler import asizeof
import nimare
import nibabel as nib 
import re
import os
import sys
import time
import gc 
import pickle
from filelock import FileLock


def get_mask(dset="1_Social_Processing"):
    dset_PATH = os.path.dirname(os.getcwd()) + '/datasets/' + dset + '/'
    Files = sorted([f for f in os.listdir(dset_PATH) if os.path.isfile(os.path.join(dset_PATH, f))]) # Filename by alphabetical order
    group_name = [re.findall('^[^_]*[^ _]', g)[0] for g in Files] # extract group name
    group_name = sorted(list(set(group_name))) # unique list
    
    group = group_name[0]
    
    MNI_file_exist = os.path.isfile(dset_PATH + group + '_MNI.txt')
    Talairach_file_exist = os.path.isfile(dset_PATH + group + '_Talairach.txt')
    if MNI_file_exist and Talairach_file_exist:
        group_dset_MNI = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_MNI.txt')
        group_dset_Talairach = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_Talairach.txt')
        group_dset = group_dset_MNI.merge(group_dset_Talairach)
    else: 
        if MNI_file_exist: 
            group_dset = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_MNI.txt') 
        elif Talairach_file_exist: 
            group_dset = nimare.io.convert_sleuth_to_dataset(text_file=dset_PATH + group + '_Talairach.txt')
    # extract mask from dataset
    mask = group_dset.masker
    n_voxel = mask.n_elements_
    del group_dset

    return mask

def get_spline_spatial_encodings(H, W, D, brain_mask=None, return_smoothness=False,
                                 spacing=10, margin=20, dtype=np.float64):
    X_spatial, bases = B_spline_bases(H, W, D, brain_mask, spacing=spacing, margin=20, dtype=dtype) # (H*W*D, P)
    if return_smoothness:
        J = smoothness_penalty(H, W, D, brain_mask, bases, spacing=spacing, margin=margin, dtype=dtype)
        return X_spatial, J
    return X_spatial


def generate_data(X, n_groups, group_names, n_studies, n_covs, n_realizations, bump_signal, use_high_intensity, brain_mask=None, dtype=np.float64):
    n_studies = [n_studies] * n_groups if isinstance(n_studies, int) else n_studies
    def generate_spatial_intensity(X, n_groups, n_studies, bump_signal):
        study_level_moderator = {group_names[i]: np.random.uniform(0.9, 1.1, size=[n_studies[i],]).astype(dtype)
                                 for i in range(n_groups)}
        if bump_signal:
            C1 = np.array([0.2, 0.2, 0.2], dtype=dtype)
            C2 = np.array([0.8, 0.8, 0.8], dtype=dtype)
            MU0 = np.exp(-np.sum((X - C1[None, :])**2, axis=-1)) \
                + np.exp(-np.sum((X - C2[None, :])**2, axis=-1)) # [H * W * D,]
            MU0 = MU0 - MU0.min() / (MU0.max() - MU0.min())
            multiplier = 1e-1 if use_high_intensity else 1e-4
            MU0 = MU0 * multiplier
        else:
            multiplier = 1e-1 if use_high_intensity else 5e-5
            MU0 = np.ones_like(X[:, 0], dtype=dtype) * multiplier
        
        if brain_mask is not None:
            # Remove intensity outside of brain mask
            brain_mask_bool = brain_mask.mask_img._dataobj.reshape(-1).astype(bool)
            MU0 = MU0[brain_mask_bool]
        
        MU = {}
        for name in group_names:
            MU_group = MU0[None, :] * study_level_moderator[name][:, None] # [N_STUDIES, H * W * D]
            MU[name] = MU_group# .astype(dtype)

        return MU, MU0
    
    MU, MU0 = generate_spatial_intensity(X, n_groups, n_studies, bump_signal) # [N_STUDIES, H * W * D]
    if brain_mask is not None:
        brain_mask_bool = brain_mask.mask_img._dataobj.reshape(-1).astype(bool)
        n_voxel = np.sum(brain_mask_bool)
    else:
        n_voxel = X.shape[0]
    Y, Z = {}, {}
    for i in range(n_groups):
        name = group_names[i]
        tmp = np.broadcast_to(MU[name][np.newaxis, ...], [n_realizations, n_studies[i], n_voxel])
        Y_group = np.random.binomial(n=1, p=tmp).astype(bool) # [N_REALIZATIONS, N_STUDIES, N_VOXEL]
        Y[name] = scipy.sparse.csr_matrix(Y_group.reshape(n_realizations, n_studies[i] * n_voxel)) # np.stack([Y_group[i] for i in range(n_realizations)])
        # Y[name] = [scipy.sparse.csr_matrix(Y_group[i]) for i in range(n_realizations)]
        Z[name] = np.random.normal(0., 0.1, size=(n_studies[i], n_covs)).astype(dtype=dtype) # [N_REALIZATIONS, N_COVS]
        logging.info(f"Group {name} has {n_studies[i]} studies")
    return MU, MU0, Y, Z

def save_to_memmap(filename, data):
    if isinstance(data, dict):
        for key, value in data.items():
            save_to_memmap(filename + f"_{key}", value)
        logging.info(f"Saved dict of ndarrays to memmap {filename}")
        return
    fp = np.memmap(filename, dtype=data.dtype, mode='w+', shape=data.shape)
    meta_data = {"shape": data.shape, "dtype": data.dtype}
    with open(filename+"_metadata", "wb") as f:
        np.save(f, meta_data)
    fp[:] = data[:]
    fp.flush()
    del fp
    gc.collect()
    logging.info(f"Saved ndarray to memmap {filename}")

def load_from_memmap(filename, dict_keys=None):
    if dict_keys is not None:
        data = {}
        for key in dict_keys:
            data[key] = load_from_memmap(filename + f"_{key}")
        logging.info(f"Loaded dict of ndarrays from memmap {filename}")
        return data
    with open(filename+"_metadata", "rb") as f:
        meta_data = np.load(f, allow_pickle=True)
    fp = np.memmap(filename, mode='r', dtype=meta_data.item()["dtype"], shape=meta_data.item()["shape"])
    data = np.array(fp)
    del fp
    gc.collect()
    logging.info(f"Loaded ndarray from memmap {filename}")
    return data

def check_job_array_status(job_array_name, lock, data_folder):
    with lock:
        with open(f'{data_folder}/jobs_{job_array_name}.pkl', 'rb') as file:
            jobs = pickle.load(file)
    num_pending, num_running, num_finished = 0, 0, 0
    for i, job in jobs.items():
        if job["status"] == "pending":
            num_pending += 1
        elif job["status"] == "running":
            num_running += 1
        elif job["status"] == "finished":
            num_finished += 1
    logging.info(f"Number of pending jobs: {num_pending}, running jobs: {num_running}, finished jobs: {num_finished}")
    return num_pending, num_running, num_finished

def pp_plot(values, log_scale=False, save_to="PP.png", alpha=0.05, lim=4.0):
    rv = np.linspace(0, 1, num=np.prod(values.shape)) 

    if not log_scale:
        xx = rv
        yy = np.sort(values)
    else:
        xx = np.sort(-np.log10(rv))
        yy = np.sort(-np.log10(values))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x=xx, y=yy, s=10)
    ax.plot([0, lim], [0, lim], 'r--')
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel("Expected p-values (-log10 scale)")
    ax.set_ylabel("Observed p-values (-log10 scale)")
    fig.savefig(save_to)
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
from util import (
    get_mask, 
    get_spline_spatial_encodings, 
    generate_data, save_to_memmap,
    load_from_memmap, 
    check_job_array_status,
    pp_plot
)
from multiprocessing import Pool 
import multiprocessing.shared_memory as sm
from memory_profiler import profile
import argparse
import pickle
from filelock import FileLock
import warnings
warnings.filterwarnings("ignore")
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for data generation, regression, and inference")

    # Boolean flags
    parser.add_argument('--run_data_generation', action='store_true', default=False,
                        help="Run data generation step (default: False)")
    parser.add_argument('--run_regression', action='store_true', default=False,
                        help="Run regression step (default: False)")
    parser.add_argument('--run_inference', action='store_true', default=False,
                        help="Run inference step (default: False)")

    # General options
    parser.add_argument('--job_array_name', type=str, default="masked_synthetic_data",
                        help="Name of the job array (default: masked_synthetic_data)")
    parser.add_argument('--num_jobs', type=int, default=10,
                        help="Number of jobs (default: 10)")
    parser.add_argument('--profile', action='store_true', default=False,
                        help="Enable profiling (default: False)")
    parser.add_argument('--use_monte_carlo', action='store_true', default=False,
                        help="Use Monte Carlo sampling (default: False)")
    parser.add_argument('--bump_signal', action='store_true', default=False,
                        help="Apply signal bump (default: False)")
    parser.add_argument('--use_high_intensity', action='store_true', default=False,
                        help="Use higher intensity for sanity check (default: False)")
    parser.add_argument('--use_penalty', action='store_true', default=False,
                        help="Apply penalty in the model (default: False)")
    parser.add_argument('--use_fi_inference', action='store_true', default=False,
                        help="Apply Fisher Information matrix for inference (default: False)")
    
    # Group related options
    parser.add_argument('--n_groups', type=int, default=2,
                        help="Number of groups (default: 2)")
    parser.add_argument('--group_names', nargs='+', default=["group_0", "group_1"],
                        help="Names of the groups (default: ['group_0', 'group_1'])")

    # Dimensional parameters
    parser.add_argument('--H', type=int, default=91,
                        help="Height of the data (default: 91)")
    parser.add_argument('--W', type=int, default=109,
                        help="Width of the data (default: 109)")
    parser.add_argument('--D', type=int, default=91,
                        help="Depth of the data (default: 91)")
    
    # Study parameters
    parser.add_argument('--n_studies', nargs='+', type=int, default=[100, 500],
                        help="Number of studies (default: [100, 500])")
    parser.add_argument('--n_realizations', type=int, default=10,
                        help="Number of realizations (default: 10)")
    parser.add_argument('--n_covs', type=int, default=2,
                        help="Number of study-level covariates (default: 2)")
    parser.add_argument('--n_resample', type=int, default=100,
                        help="Number of resamplings of studies (default: 100)")

    # Mask option
    parser.add_argument('--use_brain_mask', action='store_true', default=False,
                        help="Use brain mask (default: False)")

    # Parallel options
    parser.add_argument('--cpu_per_job', type=int, default=1,
                        help="Number of CPU per job (default: 1)")

    return parser.parse_args()

if __name__ == '__main__':

    """
    Data generation example:
        python main.py --run_data_generation --use_brain_mask --bump_signal --use_penalty --use_monte_carlo
    Regression example:
        python main.py --run_regression --num_jobs=64 --use_brain_mask --bump_signal --use_penalty --use_monte_carlo --cpu_per_job=4
    Inference example:
        python main.py --run_inference --use_brain_mask --bump_signal --use_penalty
    
    Memory profiling for data generation:
        mprof run python main.py --run_data_generation --use_brain_mask --bump_signal --use_penalty 
        mprof plot [plot filename]
    Memory profiling for regression:
        mprof run python main.py --run_regression --num_jobs=64 --use_brain_mask --bump_signal --use_penalty --cpu_per_job=4 --profile
        mprof plot [plot filename]
    """
    np.random.seed(42) # global seeding
    logging.set_verbosity(logging.INFO) # set logging level
    args = get_args()

    RUN_DATA_GENERATION = args.run_data_generation
    RUN_REGRESSION = args.run_regression
    RUN_INFERENCE = args.run_inference

    JOB_ARRAY_NAME = args.job_array_name
    NUM_JOBS = args.num_jobs
    PROFILE = args.profile
    USE_MONTE_CARLO = args.use_monte_carlo
    USE_FI_INFERENCE = args.use_fi_inference
    BUMP_SIGNAL = args.bump_signal
    USE_HIGH_INTENSITY = args.use_high_intensity
    USE_PENALTY = args.use_penalty
    N_GROUPS = args.n_groups
    GROUP_NAMES = args.group_names
    H, W, D = args.H, args.W, args.D
    N_STUDIES = args.n_studies
    N_REALIZATIONS = args.n_realizations
    N_COVS = args.n_covs # Number of study-level covariates
    N_RESAMPLE = args.n_resample # Number of resampling of studies
    USE_BRAIN_MASK = args.use_brain_mask

    filename_0 = f"_2groups_{N_STUDIES}" if N_GROUPS == 2 else "1group" if N_GROUPS == 1 else ""
    filename_1 = "_monte_carlo" if USE_MONTE_CARLO else "_resampling"
    filename_2 = "_bump_signal" if BUMP_SIGNAL else "_homo_intensity"
    job_array_lock = FileLock("job_array.lock")
    output_lock = FileLock("output.lock")

    folder_name = "high_intensity" if USE_HIGH_INTENSITY else "low_intensity"
    data_folder_name = "{}/data{}{}".format(folder_name, filename_1, filename_2)

    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)
    if not os.path.exists("output"):
        os.makedirs("output")

    if RUN_DATA_GENERATION:
        brain_mask = get_mask() if USE_BRAIN_MASK else None
        logging.info(f"Construct 3D meshgrid of size ({H}, {W}, {D})...")
        X1, X2, X3 = np.meshgrid(np.linspace(0., 1.0, num=H), 
                                np.linspace(0., 1.0, num=W),
                                np.linspace(0., 1.0, num=D),
                                indexing="ij")
        X_meshgrid = np.stack([X1, X2, X3], axis=-1) # (H, W, D, 3)
        X = X_meshgrid.reshape(-1, 3) # (H*W*D, 3)
        # Remove voxels outside of brain mask
        X_img = nib.Nifti1Image(X_meshgrid, brain_mask.mask_img.affine)
        X_brain = brain_mask.transform(X_img).T # (n_voxel, 3)
        del X1, X2, X3, X_meshgrid, X_img
        gc.collect()

        logging.info(f"Construct spatial design matrix ...")
        X_spatial, J = get_spline_spatial_encodings(H, W, D, brain_mask=brain_mask, return_smoothness=True, spacing=5, margin=20, dtype=np.float64)
        P = X_spatial.shape[1] # Number of spatial features, dimension of beta
        print(P)
        logging.info(f"Contruct spatial feature matrix with {X_spatial.shape[0]}, and {P} features")
        gc.collect()

        logging.info("Generate and save data ...")
        MU, MU0, Y, Z = generate_data(X, N_GROUPS, GROUP_NAMES, N_STUDIES, N_COVS, N_REALIZATIONS, BUMP_SIGNAL, USE_HIGH_INTENSITY, brain_mask=brain_mask, dtype=np.float16)
        save_to_memmap("{}/X_spatial".format(data_folder_name), X_spatial)
        save_to_memmap("{}/J".format(data_folder_name), J)
        save_to_memmap("{}/MU".format(data_folder_name), MU)
        save_to_memmap("{}/MU0".format(data_folder_name), MU0)
        # save_to_memmap("{}/Y".format(data_folder_name), Y)
        for group in GROUP_NAMES:
            scipy.sparse.save_npz(f"{data_folder_name}/Y_{group}.npz", Y[group])
        save_to_memmap("{}/Z".format(data_folder_name), Z)

        logging.info("Create job array ...")
        keys = list(random.split(random.PRNGKey(43), N_REALIZATIONS * N_RESAMPLE))
        jobs = {}
        job_id = 0
        for i in range(N_REALIZATIONS):
            for k in range(N_RESAMPLE):
                jobs[job_id] = {
                    "realization": i,
                    "resample": k,
                    "rng": keys.pop(),
                    "X": "{}/X_spatial".format(data_folder_name),
                    "Y": ["{data_folder_name}/Y_{group}.npz" for group in GROUP_NAMES],
                    "Z": ["{data_folder_name}/Z_{group}" for group in GROUP_NAMES],
                    "monte_carlo": USE_MONTE_CARLO,
                    "penalty": USE_PENALTY,
                    "group_names": GROUP_NAMES,
                    "n_groups": N_GROUPS,
                    "n_studies": N_STUDIES,
                    "J": "{}/J".format(data_folder_name),
                    "status": "pending" # pending, running, finished
                }
                job_id += 1
        del X_spatial, J, MU, MU0, Y, Z
        gc.collect()

        with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'wb') as file:
            pickle.dump(jobs, file)
            file.flush()
        
        with output_lock:
            with open(f"output/params_{JOB_ARRAY_NAME}_{filename_0}{filename_1}{filename_2}_{folder_name}.npz", "wb") as file:
                np.savez(file, beta_hat=np.zeros((N_REALIZATIONS, N_RESAMPLE, P*N_GROUPS)), gamma_hat=np.zeros((N_REALIZATIONS, N_RESAMPLE, N_COVS)))
        
    if RUN_REGRESSION or RUN_INFERENCE:
        logging.info("Load data from memmap ...")
        X_spatial = load_from_memmap("{}/X_spatial".format(data_folder_name), dict_keys=None)
        J = load_from_memmap("{}/J".format(data_folder_name), dict_keys=None)
        MU = load_from_memmap("{}/MU".format(data_folder_name), dict_keys=GROUP_NAMES)
        MU0 = load_from_memmap("{}/MU0".format(data_folder_name), dict_keys=None)
        # Y = load_from_memmap("{}/Y".format(data_folder_name), dict_keys=GROUP_NAMES)
        Y = {}
        for group in GROUP_NAMES:
            Y[group] = scipy.sparse.load_npz(f"{data_folder_name}/Y_{group}.npz")
        Z = load_from_memmap("{}/Z".format(data_folder_name), dict_keys=GROUP_NAMES)
        P = X_spatial.shape[1]

    if RUN_REGRESSION:
        def objective(params, X, Y_g_tp, Y_t_tp, Z_tp, penalty_term, lam=0.1):
            gamma = params[-N_COVS:]
            log_l = 0
            for i in range(N_GROUPS):
                beta_g = params[i*P:(i+1)*P]
                log_mu_spatial_g = jnp.matmul(X,beta_g)
                mu_spatial_g = jnp.exp(log_mu_spatial_g)
                Z_g = Z_tp[i]
                log_mu_covariates_g = jnp.matmul(Z_g, gamma)
                mu_covariates_g = jnp.exp(log_mu_covariates_g)
                group_log_l = jnp.sum(jnp.matmul(Y_g_tp[i], log_mu_spatial_g)) \
                                + jnp.sum(jnp.matmul(Y_t_tp[i], log_mu_covariates_g)) \
                                - jnp.sum(mu_spatial_g) * jnp.sum(mu_covariates_g) 

                group_smooth_penalty = beta_g @ penalty_term @ beta_g.T
                group_log_l = group_log_l - lam * group_smooth_penalty
                # sum log_l for each group
                log_l = log_l + group_log_l
            return -log_l

        objective = jax.jit(objective, static_argnums=(1, 2, 3, 4, 5))
        value_and_grad_fun = jax.value_and_grad(objective, argnums=0)
        value_and_grad_fun = jax.jit(value_and_grad_fun, static_argnums=(1, 2, 3, 4, 5))

        def run_regression(i, k, rng_key, X, Y, Z, monte_carlo, penalty, group_names, n_groups, n_studies, J=None):
            warnings.filterwarnings("ignore")
            jax.config.update('jax_platform_name', 'cpu')
            logging.info(f"Run regression -- realization {i}, resample {k}")
            rng_key, subkey = random.split(rng_key)
            # prepare data
            penalty_term = jnp.array(J) if penalty else None
            # X = jnp.array(X)
            Z = tuple([jnp.array(Z[group]) for group in group_names])
            
            Y_g_resample, Y_t_resample = [], []
            if monte_carlo:
                for j, group in enumerate(group_names):
                    rng_key, subkey = random.split(rng_key)
                    if k == 0:
                        Y_resample = Y[group].toarray().reshape((n_studies[j], -1))
                    else:
                        Y_resample = random.permutation(rng_key, Y[group].toarray().reshape((n_studies[j], -1)), axis=1)
                    Y_resample = jnp.array(Y_resample)
                    Y_g_resample_group = jnp.sum(Y_resample, axis=0)
                    Y_t_resample_group = jnp.sum(Y_resample, axis=1)
                    # save to dictionary
                    Y_g_resample.append(Y_g_resample_group)
                    Y_t_resample.append(Y_t_resample_group)
            else:
                for j, g in enumerate(range(n_groups)):
                    n_study_g = n_studies[g]
                    group = group_names[g]
                    rng_key, subkey = random.split(rng_key)
                    study_ids_g = jnp.arange(n_study_g) if k==0 else random.choice(subkey, jnp.arange(n_study_g), shape=(n_study_g,))
                    # extract sufficient statistics
                    Y_resample = jnp.array(Y[group].toarray().reshape((n_studies[j], -1))[study_ids_g])
                    Y_g_resample_group = jnp.sum(Y_resample, axis=0)
                    Y_t_resample_group = jnp.sum(Y_resample, axis=1)
                    # save to dictionary
                    Y_g_resample.append(Y_g_resample_group)
                    Y_t_resample.append(Y_t_resample_group)

            Y_g_resample = tuple(Y_g_resample)
            Y_t_resample = tuple(Y_t_resample)

            logging.info(f"---- Realization {i} ----")

            def gc_callback(x):
                gc.collect()
                jax.clear_caches()
            optimizer = ScipyMinimize(method='L-BFGS-B', fun=value_and_grad_fun, maxiter=5000, 
                                    tol=1e-9, dtype=jnp.float64, options={'disp': False, 'maxcor': 10}, # use maxcor history steps to construct Hessian
                                    value_and_grad=True, callback=gc_callback, jit=True) 
            start_time = time.time()
            res = optimizer.run(jnp.zeros(P*N_GROUPS+N_COVS), 
                                X, Y_g_resample, Y_t_resample, Z, penalty_term)
            logging.info(f"Time elapsed: {time.time()-start_time}")
            del Y_g_resample, Y_t_resample
            gc.collect()
            return res.params

        if PROFILE:
            run_regression = profile(run_regression)

        # _Y = {group: Y[group][0] for group in GROUP_NAMES}
        # other_args = [X_spatial, _Y, Z, USE_MONTE_CARLO, USE_PENALTY, GROUP_NAMES, N_GROUPS, N_STUDIES, J]
        # run_regression(0, 0, random.PRNGKey(43), *other_args)
        # quit()

        n_cpus = multiprocessing.cpu_count()
        logging.info(f"Total number of CPU available {n_cpus}")
        n_parallel = n_cpus // args.cpu_per_job
        logging.info(f"Number of parallel jobs {n_parallel}")
        
        check_job_array_status(JOB_ARRAY_NAME, lock=job_array_lock, data_folder=data_folder_name)
        with job_array_lock:
            with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'rb') as file:
                jobs = pickle.load(file)
            count = 0
            job_ids = []
            job_stack = []
            for job_id, job in jobs.items():
                if job["status"] == "pending":
                    _Y = {group: Y[group][job['realization']] for group in GROUP_NAMES}
                    job["status"] = "running"
                    job_ids.append(job_id)
                    job_stack.append([job["realization"], job["resample"], job["rng"], X_spatial, _Y, 
                                    Z, job["monte_carlo"], job["penalty"], job["group_names"], 
                                    job["n_groups"], job["n_studies"], J])
                    count += 1
                if count >= NUM_JOBS:
                    break

            with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'wb') as file:
                pickle.dump(jobs, file)
                file.flush()
        logging.info(f"Loaded job array with {NUM_JOBS} jobs ...")
        check_job_array_status(JOB_ARRAY_NAME, lock=job_array_lock, data_folder=data_folder_name)
 
        # keys = list(random.split(random.PRNGKey(43), N_REALIZATIONS * N_RESAMPLE))
        # job_stack = []
        # for i in range(N_REALIZATIONS):
        #     for k in range(N_RESAMPLE):
        #         args = [i, k, keys.pop()]
        #         _Y = {group: Y[group][i] for group in GROUP_NAMES}
        #         other_args = [X_spatial, _Y, Z, USE_MONTE_CARLO, USE_PENALTY, GROUP_NAMES, N_GROUPS, N_STUDIES, J]
        #         job_stack.append(args + other_args)
        try:
            results = []
            for all_args in job_stack:
                res_params = run_regression(*all_args)
                results.append(res_params)
            results = np.stack(results, axis=0)
            
            # results = Parallel(n_jobs=n_parallel, backend='loky')(delayed(run_regression)(*all_args) for all_args in job_stack)
            # results = np.array(results) 
            with output_lock:
                with open(f"output/params_{JOB_ARRAY_NAME}_{filename_0}{filename_1}{filename_2}_{folder_name}.npz", "rb") as file:
                    all_results = np.load(file)
                    beta_hat = all_results["beta_hat"]
                    gamma_hat = all_results["gamma_hat"]
                for idx, item in enumerate(job_stack):
                    i, k = item[:2]
                    beta_hat[i, k] = results[idx, :P*N_GROUPS]
                    gamma_hat[i, k] = results[idx, -N_COVS:]
                with open(f"output/params_{JOB_ARRAY_NAME}_{filename_0}{filename_1}{filename_2}_{folder_name}.npz", "wb") as file:
                    np.savez(file, beta_hat=beta_hat, gamma_hat=gamma_hat)
            with job_array_lock:
                with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'rb') as file:
                    jobs = pickle.load(file)
                for job_id in job_ids:
                    job = jobs[job_id]
                    job["status"] = "finished"
                with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'wb') as file:
                    pickle.dump(jobs, file)
                    file.flush()
            del results
            gc.collect()
            logging.info("Saving optimized regression parameters ...")
            
        except Exception as e:
            logging.error(f"Error: {e}")
            with job_array_lock:
                with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'rb') as file:
                    jobs = pickle.load(file)
                for job_id in job_ids:
                    job = jobs[job_id]
                    if job["status"] == "running":
                        job["status"] = "pending"
                with open(f'{data_folder_name}/jobs_{JOB_ARRAY_NAME}.pkl', 'wb') as file:
                    pickle.dump(jobs, file)
                    file.flush()

        finally:
            check_job_array_status(JOB_ARRAY_NAME, lock=job_array_lock, data_folder=data_folder_name)


    if RUN_INFERENCE:
        num_pending, num_running, num_finished = check_job_array_status(JOB_ARRAY_NAME, lock=job_array_lock, data_folder=data_folder_name)
        if num_pending + num_running > 0:
            logging.info(f"There are still {num_pending} pending jobs and {num_running} running jobs")
        logging.info("Load optimized regression parameters ...")
        print(f"output/params_{JOB_ARRAY_NAME}_{filename_0}{filename_1}{filename_2}_{folder_name}.npz")
        with open(f"output/params_{JOB_ARRAY_NAME}_{filename_0}{filename_1}{filename_2}_{folder_name}.npz", "rb") as file:
            all_results = np.load(file)
            beta_hat = all_results["beta_hat"]
            gamma_hat = all_results["gamma_hat"]
        gamma_hat_check = np.abs(gamma_hat).sum(axis=-1).min()
        if gamma_hat_check > 1e-9:
            logging.info(f"Check passed")
        else:
            logging.info(f"Check failed, gamma_hat still have zero values to be filled")
        
        ETA0 = jnp.log(MU0) # [H * W * D,]
        if N_GROUPS == 1:
            ETA = jnp.matmul(beta_hat,X_spatial.T) # [N_REALIZATION, N_RESAMPLE, H * W * D]
            ETA_HAT = ETA[:,0,:] # [N_REALIZATION, H * W * D]
        elif N_GROUPS == 2:
            ETA_1 = jnp.matmul(beta_hat[:,:,:P],X_spatial.T) # [N_REALIZATION, N_RESAMPLE, H * W * D]
            ETA_2 = jnp.matmul(beta_hat[:,:,P:2*P],X_spatial.T) # [N_REALIZATION, N_RESAMPLE, H * W * D]
            ETA_DIFF = ETA_1 - ETA_2 # [N_REALIZATION, N_RESAMPLE, H * W * D]
            ETA_HAT_DIFF = ETA_1[:,0,:] - ETA_2[:,0,:] # [N_REALIZATION, H * W * D]

        # Non-Parametric Bootstrap
        if USE_MONTE_CARLO:
            if N_GROUPS == 1:
                COMP = (ETA_HAT[:,None,:] >= ETA)
            elif N_GROUPS == 2:
                COMP = (ETA_HAT_DIFF[:,None,:] >= ETA_DIFF) # [N_REALIZATION, N_RESAMPLE, H * W * D]
        else:
            if N_GROUPS == 1:
                SHIFT_ETA = ETA - ETA_HAT[:, None, :]
                BIAS_ETA_HAT = ETA_HAT[:, None, :] - ETA0[None, None, :]
                COMP = (SHIFT_ETA >= BIAS_ETA_HAT).astype(float)
            elif N_GROUPS == 2:
                SHIFT_ETA_DIFF = ETA_DIFF - ETA_HAT_DIFF[:, None, :] # [N_REALIZATION, N_RESAMPLE, H * W * D]
                BIAS_ETA_HAT_DIFF = ETA_HAT_DIFF[:, None, :]
                COMP = (SHIFT_ETA_DIFF >= BIAS_ETA_HAT_DIFF).astype(float)

        p_values = COMP.sum(axis=1) / N_RESAMPLE
        sorted_p_vals = np.sort(p_values, axis=1)
        avg_p_vals = np.mean(sorted_p_vals, axis=0)

        pp_plot(avg_p_vals, log_scale=True, save_to=f"PP{filename_0}{filename_1}{filename_2}_{folder_name}.png", lim=3.0)

        # # Fisher Information Inference
        # print(USE_FI_INFERENCE, N_GROUPS)
        # exit()
        # if USE_FI_INFERENCE:
        #     if N_GROUPS == 2:
        #         MU_1, MU_2 = np.exp(ETA_1), np.exp(ETA_2) # shape: [N_REALIZATION, N_RESAMPLE, n_voxel]
        #         # FI = - X^T W X
        #         for i, j in zip(range(N_REALIZATIONS), range(N_RESAMPLE)):
        #             MU_1_ij = MU_1[i, j] # shape: (n_voxel, )   
        #             # Step 1: Multiply X by the diagonal elements of W
        #             # Broadcasting W_diag to multiply each row of X
        #             X_w = X_spatial * MU_1_ij[:, np.newaxis] # shape: [n_voxel, P]
        #             # Step 2: Compute X^T * X_w
        #             FI_matrix = np.dot(X_spatial.T, X_w)
        #             inv_FI = np.linalg.inv(FI_matrix)
        #             print(inv_FI)
        #             print(inv_FI.shape)
        #             exit()
        #         print(X_spatial.shape)
        #         print(MU_1.shape)


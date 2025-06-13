import numpy as np
import scipy
from util import get_mask, get_spline_spatial_encodings, pp_plot
import os

print(os.path.exists("X_spatial.npy"))

if not os.path.exists("X_spatial.npy"):
    brain_mask = get_mask()
    H, W, D = 91, 109, 91
    X_spatial, J = get_spline_spatial_encodings(H, W, D, brain_mask=brain_mask, return_smoothness=True, spacing=5, margin=20, dtype=np.float64)
    np.save("X_spatial.npy", X_spatial)
else:
    X_spatial = np.load("X_spatial.npy")
P = X_spatial.shape[1] # Number of spatial features, dimension of beta

with open(f"output/params_masked_synthetic_data__2groups_[100, 500]_monte_carlo_homo_intensity_high_intensity.npz", "rb") as file:
    all_results = np.load(file)
    beta_hat = all_results["beta_hat"]
    gamma_hat = all_results["gamma_hat"]
N_REALIZATION, N_RESAMPLE, _ = beta_hat.shape

N_GROUPS = 2
if N_GROUPS == 2:
    for k in range(N_REALIZATION):
        ETA_1 = np.matmul(beta_hat[k,0,:P],X_spatial.T) # [228483, 1]
        ETA_2 = np.matmul(beta_hat[k,0,P:2*P],X_spatial.T) # [228483, 1]
        ETA_DIFF = ETA_1 - ETA_2
        MU_1, MU_2 = np.exp(ETA_1), np.exp(ETA_2)
        # Step 1: Multiply X by the diagonal elements of W
        # Broadcasting W_diag to multiply each row of X
        X_w_1 = X_spatial * MU_1[:, np.newaxis] # shape: [n_voxel, P]
        # Step 2: Compute X^T * X_w
        FI_matrix_1 = np.dot(X_spatial.T, X_w_1)
        inv_FI_1 = np.linalg.inv(FI_matrix_1)
        # compute variance of eta
        weighted_X_1 = X_spatial @ inv_FI_1
        var_eta_1 = np.sum(weighted_X_1 * X_spatial, axis=1)
        del weighted_X_1, FI_matrix_1, inv_FI_1, X_w_1
        
        # For group 2
        X_w_2 = X_spatial * MU_2[:, np.newaxis] # shape: [n_voxel, P]
        # Step 2: Compute X^T * X_w
        FI_matrix_2 = np.dot(X_spatial.T, X_w_2)
        inv_FI_2 = np.linalg.inv(FI_matrix_2)
        weighted_X_2 = X_spatial @ inv_FI_2
        var_eta_2 = np.sum(weighted_X_2 * X_spatial, axis=1)
        del weighted_X_2, FI_matrix_2, inv_FI_2, X_w_2
        Z_diff = ETA_DIFF / np.sqrt(var_eta_1 + var_eta_2)
        p_diff = 1 - scipy.stats.norm.cdf(Z_diff)
        print(Z_diff)
        print(p_diff)
        print(Z_diff.shape, p_diff.shape)
        print(np.count_nonzero(p_diff<0.05))
        pp_plot(p_diff, log_scale=True, save_to=f"test.png", lim=2.0)
        exit()

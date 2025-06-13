import numpy as np
import torch 
import scipy
from util import load_from_memmap, load_dataset, process_data

DSET_NAME = "1_Social_Processing"
GROUP_NAMES =['others', 'socialcommunication', 'self', 'affiliation']
N_REALIZATIONS = 1
brain_mask, dset = load_dataset(DSET_NAME)
Y, Z, N_STUDIES = process_data(dataset=dset, group_names=GROUP_NAMES, n_realizations=N_REALIZATIONS, mask=brain_mask, covariates=True)

data_folder_name = "data/1_Social_Processing/monte_carlo"
X_spatial = load_from_memmap("{}/X_spatial".format(data_folder_name), dict_keys=None)
P = X_spatial.shape[1]
print(P)

all_Y_g, all_Y_t = {}, {}
for j, group in enumerate(GROUP_NAMES):
    Y_ = Y[group].toarray().reshape((N_STUDIES[j], -1))
    all_Y_g[group] = np.sum(Y_, axis=0)
    all_Y_t[group] = np.sum(Y_, axis=1)
del Y

def objective1(gamma_params, all_beta_params, X, all_y_g, Z=None, all_y_t=None, GROUPS=None, lambda_param=None, device='cpu'):
    # Under the assumption that Y_ij is either 0 or 1
    # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
    log_l = 0
    all_mu_spatial, all_mu_covariates = dict(), dict()
    X = torch.tensor(X, dtype=torch.float64, device=device) if not isinstance(X, torch.Tensor) else X
    gamma_params = torch.tensor(gamma_params, dtype=torch.float64, device=device) if not isinstance(gamma_params, torch.Tensor) else gamma_params
    for group in GROUPS:
        y_g = torch.tensor(all_y_g[group].reshape((-1, 1)), dtype=torch.float64, device=device)
        y_t = torch.tensor(all_y_t[group].reshape((-1, 1)), dtype=torch.float64, device=device)
        n_study_g = y_t.shape[0]
        # estimated spatial intensity
        log_mu_spatial =  X @ torch.tensor(all_beta_params[group], dtype=torch.float64, device=device)
        mu_spatial = torch.exp(log_mu_spatial)
        sum_mu_spatial = torch.exp(torch.logsumexp(log_mu_spatial, dim=0))
        all_mu_spatial[group] = mu_spatial
        # Covariate
        # mu^Z = exp(Z * gamma)
        Z_g = torch.tensor(Z[group], dtype=torch.float64, device=device)
        log_mu_covariates = Z_g @ gamma_params
        mu_covariates = torch.exp(log_mu_covariates)
        all_mu_covariates[group] = mu_covariates
        group_log_l = torch.sum(torch.mul(y_g, log_mu_spatial)) + torch.sum(torch.mul(y_t, log_mu_covariates)) \
                            - sum_mu_spatial * torch.sum(mu_covariates) 
        log_l += group_log_l

    log_l = log_l.detach().cpu().numpy()
    return -log_l

def objective2(gamma_params, all_beta_params, X, Y_g_tp, Y_t_tp, Z_tp, group_names, lam=0.1):
    gamma = gamma_params # params[-N_COVS:]
    log_l = 0
    for group in GROUP_NAMES:
        beta_g = all_beta_params[group] # params[i*P:(i+1)*P]
        log_mu_spatial_g = np.matmul(X,beta_g)
        mu_spatial_g = np.exp(log_mu_spatial_g)
        # Covariate
        Z_g = Z_tp[group]
        log_mu_covariates_g = np.matmul(Z_g, gamma)
        mu_covariates_g = np.exp(log_mu_covariates_g)
        group_log_l = np.sum(np.matmul(Y_g_tp[group], log_mu_spatial_g)) \
                        + np.sum(np.matmul(Y_t_tp[group], log_mu_covariates_g)) \
                        - np.sum(mu_spatial_g) * np.sum(mu_covariates_g) 
        # group_smooth_penalty = beta_g @ penalty_term @ beta_g.T
        # group_log_l = group_log_l - lam * group_smooth_penalty
        # sum log_l for each group
        log_l = log_l + group_log_l

    return -log_l

np.random.seed(0)
gamma_params = np.random.rand(2, 1)
all_beta_params = dict()
for group in GROUP_NAMES:
    all_beta_params[group] = np.random.rand(P, 1)

a = objective1(gamma_params, all_beta_params, X_spatial, all_Y_g, Z, all_Y_t, GROUP_NAMES, lambda_param=0.1, device='cpu')
b = objective2(gamma_params, all_beta_params, X_spatial, all_Y_g, all_Y_t, Z, GROUP_NAMES, 0.1)
print(a)
print(b)

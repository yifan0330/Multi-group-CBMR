import numpy as np
import torch 
import scipy
from util import load_from_memmap, load_dataset, process_data

DSET_NAME = "1_Social_Processing"
GROUP_NAMES =['others', 'socialcommunication', 'self', 'affiliation']
N_GROUPS = len(GROUP_NAMES)
N_REALIZATIONS = 1
brain_mask, dset = load_dataset(DSET_NAME)
n_voxel = brain_mask.n_elements_
N_STUDIES = []

data_folder_name = "data/1_Social_Processing/Poisson_model/monte_carlo"
Y = {}
for group in GROUP_NAMES:
    Y[group] = scipy.sparse.load_npz(f"{data_folder_name}/Y_{group}.npz")
    n_study_g = int(Y[group].toarray().shape[1] / n_voxel)
    N_STUDIES.append(n_study_g)
Z = load_from_memmap("{}/Z".format(data_folder_name), dict_keys=GROUP_NAMES)

# Y, Z, N_STUDIES = process_data(dataset=dset, group_names=GROUP_NAMES, n_realizations=N_REALIZATIONS, mask=brain_mask, covariates=True)
X_spatial = load_from_memmap("{}/X_spatial".format(data_folder_name), dict_keys=None)
P = X_spatial.shape[1]

all_Y_g, all_Y_t = {}, {}
for j, group in enumerate(GROUP_NAMES):
    Y_ = Y[group].toarray().reshape((N_STUDIES[j], -1))
    all_Y_g[group] = np.sum(Y_, axis=0)
    all_Y_t[group] = np.sum(Y_, axis=1)
del Y

def objective1(gamma_params, theta_params, all_beta_params, X, all_y_g, Z=None, all_y_t=None, GROUPS=None, lambda_param=None, device='cpu'):
    # Under the assumption that Y_ij is either 0 or 1
    # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
    log_l = 0
    all_mu_covariates = dict()
    X = torch.tensor(X, dtype=torch.float64, device=device) if not isinstance(X, torch.Tensor) else X
    gamma_params = torch.tensor(gamma_params, dtype=torch.float64, device=device) if not isinstance(gamma_params, torch.Tensor) else gamma_params
    i = 0
    for group in GROUPS:
        y_g = torch.tensor(all_y_g[group].reshape((-1, 1)), dtype=torch.float64, device=device)
        y_t = torch.tensor(all_y_t[group].reshape((-1, 1)), dtype=torch.float64, device=device)
        n_study_g = y_t.shape[0]
        # estimated spatial intensity
        log_mu_spatial =  X @ torch.tensor(all_beta_params[group], dtype=torch.float64, device=device)
        mu_spatial = torch.exp(log_mu_spatial)
        sum_mu_spatial = torch.exp(torch.logsumexp(log_mu_spatial, dim=0))
        # Covariate
        # mu^Z = exp(Z * gamma)
        Z_g = torch.tensor(Z[group], dtype=torch.float64, device=device)
        log_mu_covariates = Z_g @ gamma_params
        mu_covariates = torch.exp(log_mu_covariates)
        # Now the sum of NB variates are no long NB distributed (since mu_ij != mu_i'j),
        # Therefore, we use moment matching approach,
        # create a new NB approximation to the mixture of NB distributions: 
        # alpha' = sum_i mu_{ij}^2 / (sum_i mu_{ij})^2 * alpha
        theta_g = torch.tensor(theta_params[i,0], dtype=torch.float64, device=device)
        alpha_g = 100*n_study_g * torch.nn.Sigmoid()(theta_g) + 1e-8
        # After similification, we have:
        # r' = 1/alpha * sum(mu^Z_i)^2 / sum((mu^Z_i)^2)
        # p'_j = 1 / (1 + sum(mu^Z_i) / (alpha * mu^X_j * sum((mu^Z_i)^2)
        r = 1 / alpha_g * torch.sum(mu_covariates)**2 / torch.sum(mu_covariates**2)
        p = 1 / (1 + torch.sum(mu_covariates) / (alpha_g * mu_spatial * torch.sum(mu_covariates**2)))
        # excess variance parameter: alpha
        estimated_alpha_g = alpha_g * torch.sum(mu_covariates**2) / torch.sum(mu_covariates)**2
        group_log_l = torch.sum(torch.lgamma(y_g+r) - torch.lgamma(y_g+1) - torch.lgamma(r) + r*torch.log(1-p) + y_g*torch.log(p))
        log_l += group_log_l
        i += 1
    log_l = log_l.detach().cpu().numpy()
    return -log_l

def objective2(gamma_params, theta_params, all_beta_params, X, Y_g_tp, Y_t_tp, Z_tp, group_names, lam=0.1):
    gamma = gamma_params # params[-N_COVS:]
    log_l = 0
    i = 0
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    for group in GROUP_NAMES:
        theta_g = theta_params[i]
        beta_g = all_beta_params[group] # params[i*P:(i+1)*P]
        log_mu_spatial_g = np.matmul(X,beta_g)
        mu_spatial_g = np.exp(log_mu_spatial_g)
        # Covariate
        Z_g = Z_tp[group]
        n_study_g = Z_g.shape[0]
        log_mu_covariates_g = np.matmul(Z_g, gamma)
        mu_covariates_g = np.exp(log_mu_covariates_g)
        alpha_g = 100*n_study_g * sigmoid(theta_g) + 1e-8
        # After similification, we have:
        # r' = 1/alpha * sum(mu^Z_i)^2 / sum((mu^Z_i)^2)
        # p'_j = 1 / (1 + sum(mu^Z_i) / (alpha * mu^X_j * sum((mu^Z_i)^2)
        r = 1/alpha_g * np.sum(mu_covariates_g)**2 / np.sum(mu_covariates_g**2)
        p = 1 / (1 + np.sum(mu_covariates_g) / (alpha_g * mu_spatial_g * np.sum(mu_covariates_g**2)))
        p = p.flatten()
        # excess variance parameter: alpha
        estimated_alpha_g = alpha_g * np.sum(mu_covariates_g**2) / np.sum(mu_covariates_g)**2
        group_log_l = np.sum(scipy.special.loggamma(Y_g_tp[group]+r) - scipy.special.loggamma(Y_g_tp[group]+1) \
                            - scipy.special.loggamma(r) + r*np.log(1-p) + Y_g_tp[group]*np.log(p))
        log_l = log_l + group_log_l
        i = i + 1
    return -log_l

np.random.seed(0)
gamma_params = np.random.rand(2, 1)
theta_params = np.random.rand(N_GROUPS, 1)
all_beta_params = dict()
for group in GROUP_NAMES:
    all_beta_params[group] = np.random.rand(P, 1)

a = objective1(gamma_params, theta_params, all_beta_params, X_spatial, all_Y_g, Z, all_Y_t, GROUP_NAMES, lambda_param=0.1, device='cpu')
b = objective2(gamma_params, theta_params, all_beta_params, X_spatial, all_Y_g, all_Y_t, Z, GROUP_NAMES, 0.1)
print(a)
print(b)

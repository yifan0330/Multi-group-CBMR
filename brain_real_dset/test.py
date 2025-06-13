import numpy as np
from util import load_from_memmap, Intensity_map, load_dataset
import scipy 
import os

DSET_NAME = "5_Cue_Reactivity"
MODEL = "Poisson"
brain_mask, dset = load_dataset(DSET_NAME)
N_VOXEL = brain_mask.n_elements_
GROUP_NAMES = list(dset.keys())
filename_0 = "monte_carlo"
data_folder_name = "data/{}/{}_model/{}".format(DSET_NAME, MODEL, filename_0)
Y = {}
for group in GROUP_NAMES:
    Y[group] = scipy.sparse.load_npz(f"{data_folder_name}/Y_{group}.npz")
    Y[group] = Y[group].toarray().reshape((-1, N_VOXEL))

    # affiliation, 31 studies, 207 foci

# # For creating intensity map of the previous CBMR pipeline
# with open("output/params_masked_synthetic_data_1_Social_Processing_monte_carlo.npz", "rb") as file:
#     all_results = np.load(file)
#     beta_hat = all_results["beta_hat"]
#     gamma_hat = all_results["gamma_hat"]
# beta = beta_hat[:,0,:].reshape((4, 2624)) # shape: [4, 2624]  

beta_filename = f"/well/nichols/users/pra123/CBMR_neuroimage_paper/outcomes/20_dset_meta_regression/with_covariate_results/{DSET_NAME}/Poisson_model/Spline_penalty_2/lambda_0.1/spacing_5/bootstrapping/homo_init/beta.npy"
print(beta_filename)
all_beta = np.load(beta_filename)
beta = all_beta[:,0,:,:] # shape: [1, 4, 2624, 1]

print("{}/X_spatial".format(data_folder_name))
X_spatial = load_from_memmap("{}/X_spatial".format(data_folder_name), dict_keys=None)
N_VOXEL, P = X_spatial.shape # 228483, 2624

beta = beta.reshape((-1, P)) # shape: [4, 2624]
eta = beta @ X_spatial.T # shape: [4, 228483]

save_to = os.getcwd()+"/test_IntensityMap/{}/".format(DSET_NAME)
Intensity_map(param=eta, group_names=GROUP_NAMES, brain_mask=brain_mask, param_name="ETA", save_to=save_to)
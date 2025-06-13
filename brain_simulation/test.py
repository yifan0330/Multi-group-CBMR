import numpy as np
from util import get_mask, get_spline_spatial_encodings

output = np.load("output/params_masked_synthetic_data__2groups_[100, 500]_monte_carlo_bump_signal_low_intensity.npz")
# Display the list of arrays in the file
print("Available arrays in the file:", output.files)
# Access and print each array
beta_hat = output["beta_hat"] # [10,100,5248]
gamma_hat = output["gamma_hat"] # [10,100,2]

H, W, D = 91, 109, 91
X1, X2, X3 = np.meshgrid(np.linspace(0., 1.0, num=H), 
                        np.linspace(0., 1.0, num=W),
                        np.linspace(0., 1.0, num=D),
                        indexing="ij")
X_meshgrid = np.stack([X1, X2, X3], axis=-1) # (H, W, D, 3)
X = X_meshgrid.reshape(-1, 3) # (H*W*D, 3)

brain_mask = get_mask()
X_spatial, J = get_spline_spatial_encodings(H, W, D, brain_mask=brain_mask, return_smoothness=True, spacing=5, margin=20, dtype=np.float64)

C1 = np.array([0.2, 0.2, 0.2], dtype=np.float64)
C2 = np.array([0.8, 0.8, 0.8], dtype=np.float64)
MU0 = np.exp(-np.sum((X - C1[None, :])**2, axis=-1)) \
    + np.exp(-np.sum((X - C2[None, :])**2, axis=-1)) # [H * W * D,]
MU0 = MU0 - MU0.min() / (MU0.max() - MU0.min())
multiplier = 1e-4
MU0 = MU0 * multiplier

# multiplier = 5e-5 #if use_high_intensity else 5e-5
# MU0 = np.ones_like(X[:, 0], dtype=np.float64) * multiplier

brain_mask_bool = brain_mask.mask_img._dataobj.reshape(-1).astype(bool)
MU0 = MU0[brain_mask_bool]
print(MU0)
print(MU0.shape)

P = X_spatial.shape[1]
print(beta_hat.shape)
beta_1 = beta_hat[:, :, :P].transpose(0, 2, 1)  # Shape (10, P, 100)
beta_2 = beta_hat[:, :, P:].transpose(0, 2, 1)  # Shape (10, 2624-P, 100)
eta_1_hat = np.array([X_spatial @ beta_1[i] for i in range(beta_1.shape[0])])  # Shape (10, 228483, 100)
eta_2_hat = np.array([X_spatial @ beta_2[i] for i in range(beta_2.shape[0])])  # Shape (10, 228483, 100)
eta_1_hat = eta_1_hat.transpose((0,2,1))
eta_2_hat = eta_2_hat.transpose((0,2,1))

mu_1_hat = np.exp(eta_1_hat)
mu_2_hat = np.exp(eta_2_hat)

print(mu_1_hat.shape)
print(mu_2_hat.shape)
print(mu_1_hat)
print(mu_2_hat)

mu_1_bias = mu_1_hat - MU0[np.newaxis, np.newaxis, :]
mu_2_bias = mu_2_hat - MU0[np.newaxis, np.newaxis, :]

print(MU0[np.newaxis, np.newaxis, :].shape)
print(mu_1_bias.shape, mu_2_bias.shape)
print(np.mean(mu_1_bias), np.mean(mu_2_bias))
exit()

# # Check where all elements along the 3rd dimension are zero
# zero_mask = np.all(beta_hat == 0, axis=2)
# print(zero_mask)
# print(zero_mask.shape)
# # Get the indices of the elements that are all zero along the 3rd dimension
# indices_1, indices_2 = np.where(zero_mask)
# print("Indices where all elements are zero along the 3rd dimension:")
# print(indices_1)
# print(indices_2)
# print(indices_1.shape, indices_2.shape)


# import pickle
# filepath = "data__monte_carlo__bump_signal/jobs_masked_synthetic_data.pkl"
# # Open the file in read-binary mode
# with open(filepath, 'rb') as file:
#     data = pickle.load(file)

# # 'data' now contains the deserialized Python object
# # print(data[0]['status'])

# indices = [key for key, value in data.items() if data[key]['status'] == 'running']
# # print(indices)
# # print(len(indices))
# # exit()

# for i in indices:
#     data[i]['status'] = 'pending'

# # indices = [key for key, value in data.items() if data[key]['status'] == 'running']
# # print(indices)

# # Open a file in write-binary ('wb') mode to save the dictionary
# with open(filepath, 'wb') as file:
#     pickle.dump(data, file)



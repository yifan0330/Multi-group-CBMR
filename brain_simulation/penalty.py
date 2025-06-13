import numpy as np

def sp_dimension(l, brain_mask, basis, spacing, margin):
    if brain_mask is not None:
        masker_voxels = brain_mask.mask_img._dataobj
        xx = np.where(np.apply_over_axes(np.sum, masker_voxels, [1, 2]) > 0)[0]
    else:
        xx = np.arange(l)
    wider_xx = np.arange(np.min(xx) - margin, np.max(xx) + margin)
    xx_knots = np.arange(np.min(wider_xx), np.max(wider_xx), step=spacing)
    xx_knots = np.concatenate(([xx_knots[0]]*2, xx_knots, [xx_knots[-1]]*2), axis=0)
    s_x = sBsp(k=4, u=xx_knots)
    
    return s_x
    
def smoothness_penalty(H, W, D, brain_mask, bases, spacing=5, margin=20, dtype=np.float64):
    # Load the supported bases for x, y, z directions
    x_support_basis, y_support_basis, z_support_basis, support_basis = bases

    # Smoothness penalty in x, y, z directions
    s_x = sp_dimension(H, brain_mask, x_support_basis, spacing, margin)
    s_y = sp_dimension(W, brain_mask, y_support_basis, spacing, margin)
    s_z = sp_dimension(D, brain_mask, z_support_basis, spacing, margin)

    s_x = s_x.astype(dtype)
    s_y = s_y.astype(dtype)
    s_z = s_z.astype(dtype)
    
    # Tensor product of the smoothness penalty in x, y, z directions
    J = np.einsum('ab,cd,ef->acebdf', s_x, s_y, s_z).reshape(
        s_x.shape[0] * s_y.shape[0] * s_z.shape[0],
        s_x.shape[1] * s_y.shape[1] * s_z.shape[1]
    )

    # J = np.kron(np.kron(s_x, s_y), s_z)
    J = J[support_basis, :]
    J = J[:, support_basis]

    return J

def dBsp(k, u, v=None):
    n_v = len(u) - k
    I = np.arange(n_v + 1)
    dif = u[I + k - 1] - u[I]
    dif[dif == 0] = 10
        
    K = np.diag((k - 1) / dif)
       
    S = np.zeros((n_v + 1, n_v))
    for i in range(0, n_v+1):
        if i >= 1:
            S[i, i-1] = -1
        if i <= n_v-1:
            S[i, i] = 1
    D = np.dot(K, S) # dim D: n_v+1 x n_v
    if v is None:
        dv = None
    else:
        dv = np.dot(D, v)
    
    return dv, D
    
def sBsp(k, u, v=None):
    if k != 4:
        raise ValueError('This function only works with cubic splines')
   
    u = u.flatten()
    n_u = len(u)
    n_v = n_u - k
            
    _, D3 = dBsp(k, u)
    _, D2 = dBsp(k-1, u)
    D = D2 @ D3
    # This matrix formulation works only for k=4 cubic splines;
    # it exploits the fact that the second derivative of a cubic spline is
    # a linear spline
    du = np.diff(u)
    du[:k-1] = 0
    du[n_u-k:] = 0
    dif1 = np.hstack([du[1:], 0])
    dif2 = du + dif1
    dif3 = du
            
    R = np.zeros((n_v+2, n_v+2))
    for i in range(0, n_v+2):
        R[i,i] = 2 * dif2[i]
        if i >= 1:
            R[i, i-1] = dif1[i]
        if i <= n_v-1:
            R[i-1, i] = dif3[i-1]
    Q = D.T @ R @ D / 3
       
    if v is None:
        return Q
    else:
        s = v.T @ Q @ v
        ds = 2*Q @ v
        dds = 2*Q
        return s, ds, dds
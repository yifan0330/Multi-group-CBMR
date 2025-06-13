import numpy as np

def recu(t, i, k, u):
    if k == 1:
        if u[i] == u[i + 1]:
            B = np.zeros_like(t)
        else:
            B = np.logical_and(u[i] <= t, t < u[i + 1])
    else:
        B = w(t, i, k, u) * recu(t, i, k - 1, u) + (1 - w(t, i + 1, k, u)) * recu(t, i + 1, k - 1, u)
    return B
    
def w(t, i, k, u):
    if u[i] != u[i + k - 1]:
        wt = (t - u[i]) / (u[i + k - 1] - u[i])
    else:
        wt = 0
    return wt
    
def Bspline(t, k, u, v=None, ForceSup=1):
    n = len(u)
    if k + 1 > n:
        raise ValueError("u must be at least length k + 1")
    if (v is not None) and (len(v) + k != n) and (ForceSup):
        raise ValueError("{} knots requires {} control vertices".format(n, n - k))

    t = np.array(t).reshape(-1, 1)
    u = np.array(u).reshape(-1, 1)
    nBasis = n - k
    B = np.zeros((len(t), nBasis))
    iB = np.zeros(nBasis)
    for i in range(nBasis):
        B[:, i] = recu(t, i, k, u).flatten()
        iB[i] = (u[i + k] - u[i]) / k
    if n >= 2 * k:
        if ForceSup:
            bool_vec = np.logical_or(t < u[k], t >= u[n - k + 1]).flatten()
            B[bool_vec, :] = 0
    else:
        print("Insufficient knots to be a proper spline basis")
    if v is not None:
        B = np.dot(B, v.reshape(-1, 1))
    return B

def B_spline_bases(H, W, D, brain_mask, spacing=10, margin=20, dtype=np.float64):
    if brain_mask is not None:
        masker_voxels = brain_mask.mask_img._dataobj
        dim_mask = masker_voxels.shape
        # remove the blank space around the brain mask
        xx = np.where(np.apply_over_axes(np.sum, masker_voxels, [1, 2]) > 0)[0]
        yy = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 2]) > 0)[1]
        zz = np.where(np.apply_over_axes(np.sum, masker_voxels, [0, 1]) > 0)[2]
    else:
        xx = np.arange(H)
        yy = np.arange(W)
        zz = np.arange(D)
    # X direction
    wider_xx = np.arange(np.min(xx) - margin, np.max(xx) + margin)
    xx_knots = np.arange(np.min(wider_xx), np.max(wider_xx), step=spacing)
    xx_knots = np.concatenate(([xx_knots[0]]*2, xx_knots, [xx_knots[-1]]*2), axis=0)
    x_spline = Bspline(t=xx, k=4, u=xx_knots, ForceSup=1)
    x_support_basis = np.sum(x_spline, axis=0) > 0.1
    x_spline = x_spline[:, x_support_basis]   
    # Y direction
    wider_yy = np.arange(np.min(yy) - margin, np.max(yy) + margin)
    yy_knots = np.arange(np.min(wider_yy), np.max(wider_yy), step=spacing)
    yy_knots = np.concatenate(([yy_knots[0]]*2, yy_knots, [yy_knots[-1]]*2), axis=0)
    y_spline = Bspline(t=yy, k=4, u=yy_knots, ForceSup=1)
    y_support_basis = np.sum(y_spline, axis=0) > 0.1
    y_spline = y_spline[:, y_support_basis]
    # Z direction
    wider_zz = np.arange(np.min(zz) - margin, np.max(zz) + margin)
    zz_knots = np.arange(np.min(wider_zz), np.max(wider_zz), step=spacing)
    zz_knots = np.concatenate(([zz_knots[0]]*2, zz_knots, [zz_knots[-1]]*2), axis=0)
    z_spline = Bspline(t=zz, k=4, u=zz_knots, ForceSup=1)
    z_support_basis = np.sum(z_spline, axis=0) > 0.1
    z_spline = z_spline[:, z_support_basis]
    
    x_spline = x_spline.astype(dtype)
    y_spline = y_spline.astype(dtype)
    z_spline = z_spline.astype(dtype)
    # X = np.kron(np.kron(x_spline, y_spline), z_spline)
    X = np.einsum('ab,cd,ef->acebdf', x_spline, y_spline, z_spline).reshape(
        x_spline.shape[0] * y_spline.shape[0] * z_spline.shape[0],
        x_spline.shape[1] * y_spline.shape[1] * z_spline.shape[1]
    )

    if brain_mask is not None:
        n_x_spline_bases, n_y_spline_bases, n_z_spline_bases = x_spline.shape[1], y_spline.shape[1], z_spline.shape[1]
        
        # remove the voxels outside brain mask
        axis_dim = [xx.shape[0], yy.shape[0], zz.shape[0]]
        brain_voxels_index = [(z - np.min(zz))+ axis_dim[2] * (y - np.min(yy))+ axis_dim[1] * axis_dim[2] * (x - np.min(xx))
                            for x in xx for y in yy for z in zz if masker_voxels[x, y, z] == 1]
        X = X[brain_voxels_index, :]
        # remove tensor product basis that have no support in the brain
        support_basis = []
        # find and remove weakly supported B-spline bases
        for bx in range(n_x_spline_bases):
            for by in range(n_y_spline_bases):
                for bz in range(n_z_spline_bases):
                    basis_index = bz + n_z_spline_bases*by + n_z_spline_bases*n_y_spline_bases*bx
                    basis_coef = X[:, basis_index]
                    if np.max(basis_coef) >= 0.1: 
                        support_basis.append(basis_index)
        X = X[:, support_basis]
        
    else:
        support_basis = np.sum(X, axis=0) >= 0.1
        X = X[:, support_basis]
        
    bases = (x_support_basis, y_support_basis, z_support_basis, support_basis)

    return X, bases
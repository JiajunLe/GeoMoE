import numpy as np
import cv2
import torch

def estimate_pose_norm_kpts(kpts0, kpts1, thresh=1e-3, conf=0.99999):
	if len(kpts0) < 5:
		return None

	E, mask = cv2.findEssentialMat(
	kpts0, kpts1, np.eye(3), threshold=thresh, prob=conf,
	method=cv2.RANSAC)
 
	# E, mask = cv2.findEssentialMat(
	# kpts0, kpts1, np.eye(3), threshold=thresh, prob=conf,
    # method=cv2.USAC_MAGSAC)

	assert E is not None

	best_num_inliers = 0
	new_mask = mask
	ret = None
	for _E in np.split(E, len(E) / 3):
		n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
		if n > best_num_inliers:
			best_num_inliers = n
			ret = (R, t[:, 0], mask.ravel() > 0)

	return ret

def estimate_pose_from_E(kpts0, kpts1, mask, E):
    assert E is not None
    mask = mask.astype(np.uint8)
    E = E.astype(np.float64)
    kpts0 = kpts0.astype(np.float64)
    kpts1 = kpts1.astype(np.float64)
    I = np.eye(3).astype(np.float64)

    best_num_inliers = 0
    ret = None

    for _E in np.split(E, len(E) / 3):

        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, I, 1e9, mask=mask)

        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)
    return M

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

def knn_point(k, ref, query):
    """
    Args:
        ref:   Tensor, shape (B, N, D) 
        query: Tensor, shape (B, M, D) 
        k:     int                   
    Returns:
        dists: Tensor, shape (B, M, k)  L2 
        idx:   LongTensor, shape (B, M, k) 
    """
    dists = torch.cdist(query, ref)  
    dists, idx = dists.topk(k, dim=-1, largest=False, sorted=True)
    return dists, idx

def grouping_operation(features, idx):
    """
    Args:
        features: Tensor, shape (B, C, N)
        idx:      LongTensor, shape (B, M, k)
    Returns:
        grouped:  Tensor, shape (B, C, M, k)
    """
    B, C, N = features.shape
    B2, M, k = idx.shape
    features_expand = features.unsqueeze(2).expand(-1, -1, M, -1)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1, -1)
    grouped = torch.gather(features_expand, 3, idx_expand)
    return grouped

def get_knn_feats(feats, idx):
    """
    Args:
        feats:  Tensor, shape (B, F, N, 1)  
        coords: Tensor, shape (B, N, D)     
        k:      int                  
    Returns:
        Tensor, shape (B, F, N, k) 
    """
    features = feats.squeeze(-1)            # (B, F, N)
    knn_feats = grouping_operation(features, idx.transpose(1, 2))
    return knn_feats


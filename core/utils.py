import torch
import numpy as np
from multiprocessing import Pool as ThreadPool 

def tocuda(data):
	# convert tensor data in dictionary to cuda when it is a tensor
	for key in data.keys():
		if type(data[key]) == torch.Tensor:
			data[key] = data[key].cuda()
	return data

def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M
    
def torch_skew_symmetric(v):

    zero = torch.zeros_like(v[:, 0])

    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)

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


from core import mk_GeMMMapReduce, slicer, check
from itertools import product
import torch

def proj_fold(query, key, value):
    zs = query @ key.T
    z = torch.logsumexp(zs, dim=1)
    v = (zs - z[:, None]).exp() @ value
    return z, v

def proj_fold_bwd(query, key, value, a, g):
    # query : M x F
    # key   : N x F
    # value : N x D
    # h_z : M x N
    # h_v : N x D
    h_z = query @ key.T
    h_v = value
    # a_z, g_z : M
    # a_v, g_v : M x D
    a_z, a_v = a
    g_z, g_v = g
    # w : M x N
    w = (h_z - a_z[:, None]).exp()
    # z : M x N
    # v : N x D
    z = (g_z[:, None] + g_v @ h_v.T - (g_v * a_v).sum(1, keepdims=True)) * w
    v = w.T @ g_v
    return z @ key, z.T @ query, v

def binary_reduce(a, b):
    a_z, a_v = a
    b_z, b_v = b
    z = torch.logaddexp(a_z, b_z)
    v = a_v * torch.exp(a_z - z)[:, None] + b_v * torch.exp(b_z - z)[:, None]
    return z, v

def init(query, key, value):
    assert (query.shape[1] == key.shape[1])
    assert (key.shape[0] == value.shape[0])
    M, F = query.shape
    N, D = value.shape
    identity = query.new_full((M,), float('-inf')), query.new_zeros((M,D))
    return identity

def chunker(query, key, value):
    assert (query.shape[1] == key.shape[1])
    assert (key.shape[0] == value.shape[0])
    M, F = query.shape
    N, D = value.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice]),
            lambda X: (X[0][mslice], X[1][nslice], X[2][nslice])
        )


Attention = mk_GeMMMapReduce(
        'Attention',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_attention(q, k, v):
    return Attention.apply(q, k, v)[1]

def regular_attention(q, k, v):
    return (q @ k.T).softmax(1) @ v

# will run on cpu by default...
#if __name__ == '__main__':
#    M, N, D, F = 1024, 1024, 32, 32
#
#    Q = torch.randn(M, F, requires_grad=True, dtype=torch.double)
#    K = torch.randn(N, F, requires_grad=True, dtype=torch.double)
#    V = torch.randn(N, D, requires_grad=True, dtype=torch.double)
#
#    inputs = Q, K, V
#    mock = torch.randn(M, D)
#
#    check(gemmmr_attention, regular_attention, inputs, mock)

# moving to GPUs below
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda") # Or "cuda:1" for the second GPU, or manage via CUDA_VISIBLE_DEVICES
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    M, N, D, F = 1024, 1024, 32, 32

    Q = torch.randn(M, F, requires_grad=True, dtype=torch.double, device=device)
    K = torch.randn(N, F, requires_grad=True, dtype=torch.double, device=device)
    V = torch.randn(N, D, requires_grad=True, dtype=torch.double, device=device)

    inputs = Q, K, V
    mock = torch.randn(M, D, device=device) # Ensure mock tensor is also on the same device

    check(gemmmr_attention, regular_attention, inputs, mock)
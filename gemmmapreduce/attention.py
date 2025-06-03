from core import mk_GeMMMapReduce, slicer, timer, check_equality, check_speed
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
    mslices = list(slicer(M, 1024))
    nslices = list(slicer(N, 1024))
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

@torch.compile
def gemmmr_attention(q, k, v):
    return Attention.apply(q, k, v)[1]

@torch.compile
def regular_attention(q, k, v):
    return (q @ k.T).softmax(1) @ v

if __name__ == '__main__':
    M, N, D, F = 16*1024, 16*1024, 128, 128

    Q = torch.randn(M, F, requires_grad=True, dtype=torch.double, device='cuda')
    K = torch.randn(N, F, requires_grad=True, dtype=torch.double, device='cuda')
    V = torch.randn(N, D, requires_grad=True, dtype=torch.double, device='cuda')

    inputs = Q, K, V
    mock = torch.randn(M, D, device='cuda')

    check_equality(gemmmr_attention, regular_attention, inputs, mock)
    print(f'gemmr   time: {check_speed(gemmmr_attention, inputs, mock):.3f}')
    print(f'regular time: {check_speed(regular_attention, inputs, mock):.3f}')



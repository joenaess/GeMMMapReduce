from core import mk_GeMMMapReduce, slicer, check
from itertools import product
import torch
import torch.nn.functional as F

def proj_fold(x, p, q):
    return F.relu(x @ p) @ q,

def proj_fold_bwd(x, p, q, _, g):
    gv, = g
    xp = F.relu(x @ p)
    gq = xp.T @ gv
    gxp = (gv @ q.T) * (xp > 0)
    gx = gxp @ p.T
    gp = x.T @ gxp
    return gx, gp, gq

def binary_reduce(a, b):
    av, = a
    bv, = b
    return av + bv,

def init(x, p, q):
    assert (x.shape[1] == p.shape[0])
    assert (p.shape[1] == q.shape[0])
    B, M = x.shape
    M, K = p.shape
    K, N = q.shape
    identity = x.new_zeros((B,N))
    return identity,

def chunker(x, p, q):
    assert (x.shape[1] == p.shape[0])
    assert (p.shape[1] == q.shape[0])
    B, M = x.shape
    M, K = p.shape
    K, N = q.shape
    bslices = list(slicer(B, 512))
    kslices = list(slicer(K, 512))
    for bslice, kslice in product(bslices, kslices):
        yield (
            lambda A: (A[0][bslice],),
            lambda X: (X[0][bslice], X[1][:, kslice], X[2][kslice])
        )


MLP = mk_GeMMMapReduce(
        'MLP',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_mlp(x, p, q):
    return MLP.apply(x, p, q)[0]

def regular_mlp(x, p, q):
    return F.relu(x @ p) @ q

# if __name__ == '__main__':
#     B, M, N, K = 1024, 1024, 1024, 1024

#     X = torch.randn(B, M, requires_grad=True, dtype=torch.double)
#     P = torch.randn(M, K, requires_grad=True, dtype=torch.double)
#     Q = torch.randn(K, N, requires_grad=True, dtype=torch.double)

#     inputs = X, P, Q
#     mock = torch.randn(B, N)

#     check(gemmmr_mlp, regular_mlp, inputs, mock)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # Or "cuda:1" for the second GPU, or manage via CUDA_VISIBLE_DEVICES
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    B, M, N, K = 1024, 1024, 1024, 1024

    X = torch.randn(B, M, requires_grad=True, dtype=torch.double, device=device)
    P = torch.randn(M, K, requires_grad=True, dtype=torch.double, device=device)
    Q = torch.randn(K, N, requires_grad=True, dtype=torch.double, device=device)

    inputs = X, P, Q
    mock = torch.randn(B, N, device=device) # Ensure mock tensor is also on the same device

    check(gemmmr_mlp, regular_mlp, inputs, mock)

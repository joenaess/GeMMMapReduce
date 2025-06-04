from core import mk_GeMMMapReduce, slicer, check
from itertools import product
import torch

def proj_fold(pred, trg, true, tixs):
    # pred : M x D
    # trg  : N x D
    # true : M
    # tixs : N
    ps = pred @ trg.T
    p = torch.logsumexp(ps, dim=1)
    n = (ps * (true[:, None] == tixs[None, :])).sum(1)
    return p, n

def proj_fold_bwd(pred, trg, true, tixs, a, g):
    # pred: M x D
    # trg: N x D
    # true: M
    # tixs: N

    # h_p : M x N
    # h_n : M x N
    h_p = pred @ trg.T
    h_m = true[:, None] == tixs[None, :]
    
    # a_p, g_p : M
    # a_n, g_n : M
    a_p, a_n = a
    g_p, g_n = g

    # p: M x N
    # n: M x N
    p = g_p[:, None] * (h_p - a_p[:, None]).exp()
    n = g_n[:, None]

    gh = (p + h_m * n)
    return gh @ trg, gh.T @ pred

def binary_reduce(a, b):
    a_p, a_n = a
    b_p, b_n = b
    p = torch.logaddexp(a_p, b_p)
    n = a_n + b_n
    return p, n

def init(pred, trg, truth, tixs):
    assert (pred.shape[1] == trg.shape[1])
    assert (pred.shape[0] == truth.shape[0])
    M, D = pred.shape
    identity = pred.new_full((M,), float('-inf')), pred.new_zeros((M,))
    return identity

def chunker(pred, trg, truth, tixs):
    assert (pred.shape[1] == trg.shape[1])
    assert (pred.shape[0] == truth.shape[0])
    M, _ = pred.shape
    N, _ = trg.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice]),
            lambda X: (X[0][mslice], X[1][nslice], X[2][mslice], X[3][nslice])
        )

XEntropy = mk_GeMMMapReduce(
        'XEntropy',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_xentropy(p, t, c):
    p, n = XEntropy.apply(p, t, c, torch.arange(t.shape[0], device=t.device))
    return p - n

def regular_xentropy(p, t, c):
    return torch.nn.functional.cross_entropy(p @ t.T, c, reduction='none')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # Or "cuda:1" for the second GPU, or manage via CUDA_VISIBLE_DEVICES
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    M, N, D = 1024, 1024, 128

    pred = torch.randn(M, D, requires_grad=True, dtype=torch.double, device=device)
    trg = torch.randn(N, D, requires_grad=True, dtype=torch.double, device=device)
    true = torch.randint(N, (M,), device=device)

    inputs = pred, trg, true
    mock = torch.randn(M, device=device)

    check(gemmmr_xentropy, regular_xentropy, inputs, mock)


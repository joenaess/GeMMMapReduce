from core import mk_GeMMMapReduce, slicer, check
from itertools import product
import torch

def proj_fold(pred, trg):
    # pred : M x D
    # trg  : N x D
    # returns: M
    ps = pred @ trg.T
    p = torch.logsumexp(ps, dim=1)
    n = ((ps - p.unsqueeze(1)).exp() * ps).sum(1)
    return p, n

def proj_fold_bwd(pred, trg, a, g):
    # not implemented yet.
    return pred, trg

def binary_reduce(a, b):
    a_p, a_n = a
    b_p, b_n = b
    p = torch.logaddexp(a_p, b_p)
    n = a_n * (a_p - p).exp() + b_n * (b_p - p).exp() 
    return p, n

def init(pred, trg):
    assert (pred.shape[1] == trg.shape[1])
    M, D = pred.shape
    identity = pred.new_full((M,), float('-inf')), pred.new_zeros((M,))
    return identity

def chunker(pred, trg):
    assert (pred.shape[1] == trg.shape[1])
    M, _ = pred.shape
    N, _ = trg.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice]),
            lambda X: (X[0][mslice], X[1][nslice])
        )

Entropy = mk_GeMMMapReduce(
        'Entropy',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_entropy(p, t):
    p, n = Entropy.apply(p, t)
    return p - n

def regular_entropy(p, t):
    p = (p @ t.T).softmax(dim=1)
    return -(p * p.log()).sum(1)

if __name__ == '__main__':
    M, N, D = 1024, 1024, 128

    pred = torch.randn(M, D, requires_grad=True, dtype=torch.double)
    trg = torch.randn(N, D, requires_grad=True, dtype=torch.double)

    inputs = pred, trg
    mock = torch.randn(M)

    check(gemmmr_entropy, regular_entropy, inputs, mock)

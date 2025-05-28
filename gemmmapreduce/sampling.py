from core import mk_GeMMMapReduce, slicer, timer, check_equality
from itertools import product
import torch

gumbler = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

def proj_fold(pred, trg, tixs):
    # pred : M x D
    # trg  : N x D
    # tixs : N
    # ps   : M x N
    ps = pred @ trg.T
    # candidates : M
    candidates = (ps + gumbler.sample(ps.shape)).argmax(dim=1)
    log_weights = ps[torch.arange(len(candidates)), candidates]
    return log_weights, ps.logsumexp(dim=1), tixs[candidates]

def proj_fold_bwd(pred, trg, tixs, a, g):
    return None,

def binary_reduce(a, b):
    # a: M
    a_z, a_tz, a_c = a
    b_z, b_tz, b_c = b
    ab_z = torch.logaddexp(a_z, b_z)
    ab_tz = torch.logaddexp(a_tz, b_tz)
    sample = torch.rand(a_z.shape)
    z = torch.where(sample < (a_z - ab_z).exp(), a_z, b_z)
    c = torch.where(sample < (a_z - ab_z).exp(), a_c, b_c)
    return z, ab_tz, c

def init(pred, trg, tixs):
    assert (pred.shape[1] == trg.shape[1])
    M, _ = pred.shape
    identity = pred.new_full((M,), float('-inf')), pred.new_full((M,), float('-inf')), pred.new_zeros((M,), dtype=torch.long)
    return identity

def chunker(pred, trg, tixs):
    assert (pred.shape[1] == trg.shape[1])
    M, _ = pred.shape
    N, _ = trg.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice], A[2][mslice]),
            lambda X: (X[0][mslice], X[1][nslice], X[2][nslice])
        )

Sampler = mk_GeMMMapReduce(
        'Sampler',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_sampler(p, t):
    z, tz, c = Sampler.apply(p, t, torch.arange(t.shape[0], dtype=torch.long))
    return (z - tz).exp(), c

def regular_sampler(p, t):
    ps = (p @ t.T)
    return (ps + gumbler.sample(ps.shape)).argmax(dim=1)

if __name__ == '__main__':
    M, N, D = 1024, 1024, 1024

    pred = torch.randn(32, 1, dtype=torch.double)
    trg = torch.randn(2, 1, dtype=torch.double)

    
    with torch.no_grad():
        print(gemmmr_sampler(pred, trg))

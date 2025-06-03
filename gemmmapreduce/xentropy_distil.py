from core import mk_GeMMMapReduce, slicer, timer, check_equality
from itertools import product
import torch

def proj_fold(s_pred, s_trg, t_pred, t_trg):
    qs = s_pred @ s_trg.T
    ps = t_pred @ t_trg.T

    q = torch.logsumexp(qs, dim=1)
    p = torch.logsumexp(ps, dim=1)
    n = ((ps - p[:, None]).exp() * qs).sum(1)
    return q, p, n

def proj_fold_bwd(s_pred, s_trg, t_pred, t_trg, a, g):
    aq, ap, an = a
    gq, gp, gn = g

    qs = s_pred @ s_trg.T
    ps = t_pred @ t_trg.T
    
    Sp = (ps - ap[:, None]).exp()
    Sq = (qs - aq[:, None]).exp()

    dq = gq[:, None] * Sq + gn[:, None] * Sp
    dp = (gp[:, None] + gn[:, None] * (qs - an[:, None])) * Sp

    return dq @ s_trg, dq.T @ s_pred, dp @ t_trg, dp.T @ t_pred

def binary_reduce(a, b):
    a_q, a_p, a_n = a
    b_q, b_p, b_n = b
    p = torch.logaddexp(a_p, b_p)
    q = torch.logaddexp(a_q, b_q)
    n = a_n * (a_p - p).exp() + b_n * (b_p - p).exp()
    return q, p, n

def init(s_pred, s_trg, t_pred, t_trg):
    assert (s_pred.shape[1] == s_trg.shape[1])
    assert (t_pred.shape[1] == t_trg.shape[1])
    assert (s_pred.shape[0] == t_pred.shape[0])
    assert (s_trg.shape[0] == t_trg.shape[0])
    M, _ = s_pred.shape
    identity = s_pred.new_full((M,), float('-inf')), s_pred.new_full((M,), float('-inf')), s_pred.new_zeros((M,))
    return identity

def chunker(s_pred, s_trg, t_pred, t_trg):
    M, _ = s_pred.shape
    N, _ = s_trg.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice], A[2][mslice]),
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

def gemmmr_xentropy(s_p, s_t, t_p, t_t):
    q, p, n = XEntropy.apply(s_p, s_t, t_p, t_t)
    return q - n

def regular_xentropy(s_p, s_t, t_p, t_t):
    s_d = (s_p @ s_t.T).log_softmax(dim=1)
    t_d = (t_p @ t_t.T).softmax(dim=1)
    return (t_d * s_d).sum(1).neg()

if __name__ == '__main__':
    M, N, D = 1024*8, 1024*8, 256

    s_pred = torch.randn(M, D, requires_grad=True, dtype=torch.double)
    s_trg = torch.randn(N, D, requires_grad=True, dtype=torch.double)
    t_pred = torch.randn(M, D*2, requires_grad=True, dtype=torch.double)
    t_trg = torch.randn(N, D*2, requires_grad=True, dtype=torch.double)
    inputs = s_pred, s_trg, t_pred, t_trg
    mock = torch.randn(M)

    check_equality(gemmmr_xentropy, regular_xentropy, inputs, mock)


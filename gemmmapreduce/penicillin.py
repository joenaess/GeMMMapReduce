import os
import torch
import triton
from triton import language as tl
from triton.language.extra import libdevice
from torch.autograd import Function
from torch.autograd.function import once_differentiable

@triton.jit
def dot(a, b, acc=None, to_f32=False):
    if to_f32:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    if acc is not None:
        return tl.dot(a, b, acc=acc, input_precision='ieee')
    else:
        return tl.dot(a, b, input_precision='ieee')

@triton.jit
def proj_fold_acc_kernel(
        pred_ptr, 
        trg_ptr, 
        true_ptr, 
        p_ptr,
        n_ptr, 
        lpmax_ptr,
        m_lock_ptr,
        M, N, D,
        N_GROUP: tl.constexpr,
        M_BLOCK: tl.constexpr, 
        N_BLOCK: tl.constexpr, 
        D_BLOCK: tl.constexpr):

    m0 = tl.program_id(0) * M_BLOCK
    n0 = tl.program_id(1) * N_BLOCK * N_GROUP
    mixs = m0 + tl.arange(0, M_BLOCK)
    mmask = mixs < M

    true = tl.load(
            true_ptr + mixs,
            mmask,
            other=-1
            )

    p = tl.full((M_BLOCK,), float('-inf'), tl.float32)
    n = tl.full((M_BLOCK,), 0.0, tl.float32)
    lpmax = tl.full((M_BLOCK,), float('-inf'), tl.float32)

    for ng in tl.range(0, N_GROUP):
        ng0 = n0 + ng*N_BLOCK 
        nixs = ng0 + tl.arange(0, N_BLOCK)
        nmask = nixs < N

        ps = tl.zeros((M_BLOCK, N_BLOCK), tl.float32)

        # Compute local slice of gemm
        for d0 in tl.range(0, D, D_BLOCK):
            dixs = d0 + tl.arange(0, D_BLOCK)
            dmask = dixs < D
            pred = tl.load(
                    pred_ptr + mixs[:, None]*D + dixs[None, :],
                    mmask[:, None] & dmask[None, :],
                    other=0.0
                    )
            trg = tl.load(
                    trg_ptr + nixs[None, :]*D + dixs[:, None],
                    nmask[None, :] & dmask[:, None],
                    other=0.0
                    )
            ps = dot(pred, trg, acc=ps)

        tl.debug_barrier()

        ps = tl.where(nmask[None, :] & mmask[:, None], ps, float('-inf'))

        # Fold local results into global results
        n = n + tl.sum(
                tl.where(
                    true[:, None] == nixs[None, :],
                    ps,
                    0.0
                ),
                axis=1
            )

        hi = tl.max(ps, axis=1)
        lpmax = tl.maximum(hi, lpmax)
        hi = tl.maximum(hi, p)
        sumexp = tl.sum(tl.exp(ps - hi[:, None]), axis=1)
        p = tl.where(
                hi > float('-inf'),
                hi + tl.log(sumexp + tl.exp(p-hi)),
                float('-inf')
                )

        tl.debug_barrier()

    # Add local aggregates to global.
    tl.atomic_add(n_ptr + mixs, n, mask=mmask)
    tl.atomic_max(lpmax_ptr + mixs, lpmax, mask=mmask)
    my_lock = m_lock_ptr + m0
    while tl.atomic_cas(my_lock, 0, 1) != 0:
        pass # spin
    # LOAD
    old_p = tl.load(
            p_ptr + mixs,
            mmask,
            other=0.0
            )
    # LOGADDEXP
    hi = tl.maximum(old_p, p)
    lo = tl.minimum(old_p, p)
    new_p = tl.where(
            hi > float('-inf'),
            hi + libdevice.log1p(tl.exp(lo-hi)),
            float('-inf'),
            )
    # STORE
    tl.store(p_ptr + mixs, new_p, mask=mmask)
    tl.debug_barrier()
    tl.atomic_xchg(my_lock, 0)

@triton.jit
def proj_fold_bwd_kernel(
        pred_ptr, trg_ptr, true_ptr,
        g_pred_ptr, g_trg_ptr,
        a_p_ptr,
        g_p_ptr, g_n_ptr,
        M, N, D,
        M_BLOCK: tl.constexpr, 
        N_BLOCK: tl.constexpr, 
        D_BLOCK: tl.constexpr):

    m0 = tl.program_id(0) * M_BLOCK
    mixs = m0 + tl.arange(0, M_BLOCK)
    mmask = mixs < M

    n0 = tl.program_id(1) * N_BLOCK
    nixs = n0 + tl.arange(0, N_BLOCK)
    nmask = nixs < N

    true = tl.load(
            true_ptr + mixs,
            mmask,
            other=-1
            )

    ps = tl.zeros((M_BLOCK, N_BLOCK), tl.float32)

    for d0 in tl.range(0, D, D_BLOCK):
        dixs = d0 + tl.arange(0, D_BLOCK)
        dmask = dixs < D
        pred = tl.load(
                pred_ptr + mixs[:, None]*D + dixs[None, :],
                mmask[:, None] & dmask[None, :],
                other=0.0
                )
        trg = tl.load(
                trg_ptr + nixs[None, :]*D + dixs[:, None],
                nmask[None, :] & dmask[:, None],
                other=0.0
                )
        ps = dot(pred, trg, acc=ps)
    
    a_p = tl.load(
            a_p_ptr + mixs,
            mmask,
            other=0.0,
            )

    g_p = tl.load(
            g_p_ptr + mixs,
            mmask,
            other=0.0,
            )

    g_n = tl.load(
            g_n_ptr + mixs,
            mmask,
            other=0.0,
            )

    tl.debug_barrier()
    # M x N
    gh = g_p[:, None] * tl.exp(ps - a_p[:, None])
    gh = gh + tl.where(
        true[:, None] == nixs[None, :],
        g_n[:, None],
        0.0)

    for d0 in tl.range(0, D, D_BLOCK):
        dixs = d0 + tl.arange(0, D_BLOCK)
        dmask = dixs < D
        pred = tl.load(
                pred_ptr + mixs[:, None]*D + dixs[None, :],
                mmask[:, None] & dmask[None, :],
                other=0.0
                )

        gt = dot(gh.trans(), pred, to_f32=True)

        tl.atomic_add(
            g_trg_ptr + nixs[:, None] * D + dixs[None, :],
            gt,
            mask=nmask[:, None] & dmask[None, :]
        )

        trg = tl.load(
                trg_ptr + nixs[:, None]*D + dixs[None, :],
                nmask[:, None] & dmask[None, :],
                other=0.0
                )

        gp = dot(gh, trg, to_f32=True)

        tl.atomic_add(
            g_pred_ptr + mixs[:, None] * D + dixs[None, :],
            gp,
            mask=mmask[:, None] & dmask[None, :]
        )



@torch.compile(fullgraph=True)
def xentropy_fwd(pred, trg, true):
    M, D = pred.shape
    N, _ = trg.shape

    p = pred.new_full((M,), float('-inf'), dtype=torch.float)
    n = pred.new_zeros((M,), dtype=torch.float)
    lpmax = pred.new_full((M,), float('-inf'), dtype=torch.float)
    m_lock = pred.new_zeros((M,), dtype=torch.int32)

    grid = lambda meta: (
            triton.cdiv(M, meta['M_BLOCK']),
            triton.cdiv(N, meta['N_BLOCK'] * meta['N_GROUP'])
            )

    proj_fold_acc_kernel[grid](
            pred, trg, true, 
            p, n, lpmax,
            m_lock,
            M, N, D,
            4, 32, 64, 128, 
            )
    return p, n, lpmax

@torch.compile(fullgraph=True)
def xentropy_bwd(pred, trg, true, a_p, g_a_p, g_a_n):
    M, D = pred.shape
    N, _ = trg.shape

    grid = lambda meta: (
            triton.cdiv(M, meta['M_BLOCK']),
            triton.cdiv(N, meta['N_BLOCK'])
            )

    g_pred = pred.new_zeros(pred.shape, dtype=torch.float)
    g_trg = trg.new_zeros(trg.shape, dtype=torch.float)

    proj_fold_bwd_kernel[grid](
            pred, trg, true,
            g_pred, g_trg,
            a_p, g_a_p, g_a_n,
            M, N, D,
            64, 64, 64,
            )
    return g_pred.to(pred.dtype), g_trg.to(trg.dtype)

class XEntropyMax(torch.autograd.Function):
    @staticmethod
    def forward(pred, trg, true):
        return xentropy_fwd(pred, trg, true)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        p, _, _ = outputs 
        ctx.save_for_backward(*inputs, p)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_a_p, g_a_n):
        pred, trg, true, a_p = ctx.saved_tensors
        g_pred, g_trg = xentropy_bwd(pred, trg, true, a_p, g_a_p, g_a_n)
        return g_pred, g_trg, None

def xentropy_max(p, t, c):
    ap, an, lpmax = XEntropyMax.apply(p, t, c)
    with torch.no_grad():
        pmax = (lpmax - ap).exp()
    return (ap - an).to(p.dtype), pmax.to(p.dtype)

def regular_xentropy_max(p, t, c):
    logits = p @ t.T
    xentropy = torch.nn.functional.cross_entropy(logits, c, reduction='none')
    maxp = (logits.max(dim=1).values - logits.logsumexp(dim=1)).exp()
    return xentropy, maxp

if __name__ == '__main__':
    M, N, D = 1024+7, 1024+3, 128+1
    
    torch.manual_seed(0x5eed)
    device = 'cuda:0'
    dtype = torch.float32

    pred = torch.randn(M, D, requires_grad=True, device=device, dtype=dtype)
    trg = torch.randn(N, D, requires_grad=True, device=device, dtype=dtype)

    with torch.no_grad():
        pred *= D ** -.5
        trg *= D ** -.5

    true = torch.randint(N, (M,), device=device) % 2
    
    print('regular')
    print(xentropy_max(pred, trg, true))
    print(regular_xentropy_max(pred, trg, true))

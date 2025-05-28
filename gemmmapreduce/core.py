import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import itertools
import time
from contextlib import contextmanager

@contextmanager
def timer(fmt='duration: {:.4f}'):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    print(fmt.format(duration))

def slicer(tot, chunk=1):
    start = 0
    while (start < tot):
        end = min(tot, start + chunk)
        yield slice(start, end)
        start = end

def mk_GeMMMapReduce(
    class_name,
    *,
    init,
    chunker,
    proj_fold,
    proj_fold_bwd,
    binary_reduce,
    ):
    class DynamicFunction(torch.autograd.Function):
        @staticmethod
        def forward(*X):
            A = init(*X)
            for aslice, xslice in chunker(*X):
                a = aslice(A)
                x = xslice(X)
                local_a = proj_fold(*x)
                new_a = binary_reduce(a, local_a)
                for view, new_val in zip(a, new_a):
                    view.copy_(new_val)
            return A
    
        @staticmethod
        def setup_context(ctx, inputs, outputs):
            ctx.num_inputs = len(inputs)
            ctx.save_for_backward(*inputs, *outputs)
    
        @staticmethod
        @once_differentiable
        def backward(ctx, *gA):
            X = ctx.saved_tensors[:ctx.num_inputs]
            A = ctx.saved_tensors[ctx.num_inputs:]
            gX = [p.new_zeros(p.shape) for p in X]
            for aslice, xslice in chunker(*X):
                # Extract chunks
                x = xslice(X)
                a = aslice(A)
                ga = aslice(gA)
                # recompute and calculate local gradients.
                gx = proj_fold_bwd(*x, a, ga)
                # Add local gradients to global.
                for g, d in zip(xslice(gX), gx):
                    g.add_(d)
            return tuple(gX)
    
    DynamicFunction.__name__ = class_name
    DynamicFunction.__qualname__ = class_name
    DynamicFunction.__module__ = getattr(init, '__module__', __name__)
    
    return DynamicFunction


def check_equality(f1, f2, inputs, mock):
    for p in inputs:
        if p.grad is not None:
            p.grad.zero_()
    y1 = f1(*inputs)
    (y1 * mock).sum().backward()
    y1 = y1.detach()
    g1 = []
    for p in inputs:
        if p.grad is not None:
            g1.append(p.grad.detach().clone())
    for p in inputs:
        if p.grad is not None:
            p.grad.zero_()
    y2 = f2(*inputs)
    (y2 * mock).sum().backward()
    y2 = y2.detach()
    g2 = []
    for p in inputs:
        if p.grad is not None:
            g2.append(p.grad.detach().clone())

    print(f'{" output ":=^30}')
    delta = (y1 - y2)
    print(f'{" shapes match": <20}: {y1.shape == y2.shape}')
    print(f'{" all close": <20}: {torch.allclose(y1, y2)}')
    print(f'{" l2 diff": <20}: {delta.pow(2).sum().sqrt()}')
    print(f'{" max_diff": <20}: {delta.abs().max()}')


    print(f'{" grad ":=^30}')
    for i, (a, b) in enumerate(zip(g1, g2)):
        name = f' grad_{i} '
        print(f'  {name:-^26}  ')
        delta = a - b
        print(f'{" shapes match": <20}: {a.shape == b.shape}')
        print(f'{" all close": <20}: {torch.allclose(a, b)}')
        print(f'{" l2 diff": <20}: {delta.pow(2).sum().sqrt()}')
        print(f'{" max_diff": <20}: {delta.abs().max()}')


def check_speed(f1, inputs, mock, runs=10, warmup=3):
    for i in range(warmup):
        for p in inputs:
            if p.grad is not None:
                p.grad.zero_()
        y1 = f1(*inputs)
        (y1 * mock).sum().backward()

    start = time.perf_counter()
    for i in range(runs):
        y1 = f1(*inputs)
        (y1 * mock).sum().backward()

    return (time.perf_counter() - start) / runs
    


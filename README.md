# GeMMMapReduce

This repo is a Proof of Concept of a general framework to construct and generalize 
memory efficient layers in neural networks. The idea is the same as Map Reduce, 
or monoidal folds, namely that we identify a mapping from inputs to monoidal 
values that when folded results in an aggregate from which the output can be
produced. 

A notable example is attention, where the monoid corresponds to logarithmic
weights and vectors representing a weighted average.

```
def binary_reduce(a, b):
    # a, b : Attention Monoids represented as tuples of tensors
    # returns: Monoid
    # z: logarithmic weight (M-tensor)
    # v: weighted average (M x D-tensor)
    a_z, a_v = a
    b_z, b_v = b
    z = torch.logaddexp(a_z, b_z)
    v = a_v * torch.exp(a_z - z)[:, None] + b_v * torch.exp(b_z - z)[:, None]
    return z, v

def proj_fold(query, key, value):
    # fused projection from inputs (query, key, value)
    # and fold over dim=1, corresponding to
    # folding using the above binary_reduce function. 

    # query: M x F
    # key  : N x F
    # value: N x D
    # returns (M, M x D)

    zs = query @ key.T
    z = zs.logsumexp(dim=1)
    v = (zs - z[:, None]).exp() @ value
    return z, v
```

The above functions combined with appropriate slicing over the input values (query, key, value) and aggregate monoidal values 
results in an M-tensor of monoid values `A`, where `softmax(query, key.T, dim=1) @ value` corresponds to `A[1]`, i.e. the weighted average.

The repo contains a factory function `mk_GeMMMapReduce` which takes functions `init`, `chunker`, `proj_fold`, 
`proj_fold_bwd`, and `binary_reduce`, which performs the monoidal aggregation of results without realizing the full intermediate matrix needed
to compute the aggregate, and a similar backwards pass. As well as implementations for two-layer MLPs, Attention, XEntropy, Entropy, and Sampling.

For more information read this paper ...

The functions produced by `mk_GeMMMapReduce`, while sometimes proving more efficient than their regular torch counterparts on CPUs, are not
particularly competetive throughput wise when running on GPUs. Future work includes adapting the framework to work with user-provided
triton kernels, which hopefully will improve performance.

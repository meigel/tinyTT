"""
Autograd helpers for tinygrad-backed TT tensors.
"""

from __future__ import annotations

import tinytt._backend as tn
from tinytt import TT


def _backward(x):
    """Backward, retrying with ``retain_graph=True`` on PyTorch if needed.

    PyTorch frees the graph after ``.backward()`` by default; calling it
    twice on the same graph requires ``retain_graph=True``.  tinygrad does
    not accept that keyword, so we fall back on a ``TypeError``.
    """
    try:
        x.backward(retain_graph=True)
    except TypeError:
        x.backward()


def watch(tens, core_indices=None):
    if core_indices is None:
        for c in tens.cores:
            c.requires_grad_(True)
    else:
        for i in core_indices:
            tens.cores[i].requires_grad_(True)


def watch_list(tensors):
    for t in tensors:
        for c in t.cores:
            c.requires_grad_(True)


def unwatch(tens):
    for c in tens.cores:
        c.requires_grad_(False)


def grad(val, tens, core_indices=None):
    _backward(val)
    if core_indices is None:
        return [c.grad for c in tens.cores]
    return [tens.cores[idx].grad for idx in core_indices]


def grad_list(val, tensors, all_in_one=True):
    _backward(val)
    cores_list = []
    if all_in_one:
        for t in tensors:
            cores_list += [c.grad for c in t.cores]
    else:
        for t in tensors:
            cores_list.append([c.grad for c in t.cores])
    return cores_list

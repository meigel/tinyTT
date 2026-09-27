"""Autograd helpers for TT tensors.

Cores are ordinary ``torch.Tensor`` objects, so autograd reaches them
directly; these are conveniences for watching and collecting the gradients of
a whole TT at once.
"""

from __future__ import annotations


def _backward(x):
    """Backward, keeping the graph so a second call on it still works.

    PyTorch frees the graph after ``.backward()`` by default, which makes a
    second gradient from the same expression fail; ``retain_graph=True``
    keeps it.  This used to be wrapped in ``try/except TypeError`` because the
    tinygrad backend did not accept the keyword -- that backend went in 0.5,
    and the fallback was dead code from then on.
    """
    x.backward(retain_graph=True)


def watch(tens, core_indices=None):
    """Enable gradient tracking on all (or selected) cores of a TT tensor.

    Parameters
    ----------
    tens : TT
        The TT tensor whose cores should be watched.
    core_indices : list[int] or None
        If ``None``, all cores are watched. Otherwise only the cores at
        the given indices.
    """
    if core_indices is None:
        for c in tens.cores:
            c.requires_grad_(True)
    else:
        for i in core_indices:
            tens.cores[i].requires_grad_(True)


def watch_list(tensors):
    """Enable gradient tracking on cores of multiple TT tensors."""
    for t in tensors:
        for c in t.cores:
            c.requires_grad_(True)


def unwatch(tens):
    """Disable gradient tracking on all cores of a TT tensor."""
    for c in tens.cores:
        c.requires_grad_(False)
        c.grad = None


def grad(val, tens, core_indices=None):
    """Back-propagate through ``val`` and return core gradients.

    Parameters
    ----------
    val : Tensor
        Scalar loss value (output of a differentiable computation).
    tens : TT
        The TT tensor whose core gradients are requested.
    core_indices : list[int] or None
        If ``None``, gradients for all cores are returned. Otherwise only
        the cores at the given indices.

    Returns
    -------
    list[Tensor]
        List of gradient tensors, one per core (or per ``core_indices``).
    """
    _backward(val)
    if core_indices is None:
        return [c.grad for c in tens.cores]
    return [tens.cores[idx].grad for idx in core_indices]


def grad_list(val, tensors, all_in_one=True):
    """Back-propagate and return core gradients for multiple TT tensors.

    Parameters
    ----------
    val : Tensor
        Scalar loss value.
    tensors : list[TT]
        List of TT tensors whose core gradients are requested.
    all_in_one : bool
        If ``True``, return a flat list of all core gradients.
        If ``False``, return a list of lists, one per TT tensor.

    Returns
    -------
    list[Tensor] or list[list[Tensor]]
        Core gradients (structure depends on ``all_in_one``).
    """
    _backward(val)
    cores_list = []
    if all_in_one:
        for t in tensors:
            cores_list += [c.grad for c in t.cores]
    else:
        for t in tensors:
            cores_list.append([c.grad for c in t.cores])
    return cores_list

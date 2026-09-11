"""
Additional operations.

@author: ion
"""
import numpy as np

import tinytt._backend as tn


def apply_mask(cores, indices):
    """Evaluate a TT at a batch of multi-indices.

    Parameters
    ----------
    cores : list of Tensor
        TT cores, shape ``(r_{k-1}, n_k, r_k)``.
    indices : Tensor or array-like, shape ``(M, d)``
        One multi-index per row.

    Returns
    -------
    Tensor, shape ``(M,)``
    """
    d = len(cores)
    idx = tn.to_numpy(indices) if tn.is_tensor(indices) else np.asarray(indices)
    idx = np.asarray(idx, dtype=np.int64).reshape(len(idx), -1)
    if idx.shape[1] != d:
        raise ValueError(
            f"indices has {idx.shape[1]} columns but the TT has {d} modes"
        )
    result = tn.ones((idx.shape[0], 1), dtype=cores[0].dtype,
                     device=cores[0].device)
    for i in range(d):
        # One host transfer for the whole index array, not one per element.
        result = tn.einsum('ij,jik->ik', result,
                           cores[i][:, idx[:, i].tolist(), :])
    return tn.squeeze(result)


def dense_matvec(cores, other):
    """
    Performs multiplication between a TT-matrix and a full tensor.
    Compatible to trailing dimensions broadcasting.

    Args:
        cores (list[Tensor]): the TT-cores of the TT-matrix. The TT-matrix should be of shape (M1 x ... x Md) x (N1 x ... x Nd).
        other (Tensor): The tensor with shape B1 x ... x Bn x N1 x ... x Nd.

    Returns:
        Tensor: The result. Shape is B1 x ... x Bn x M1 x ... x Md.
    """

    def _dense_matvec_impl(other, *cores):
        result = tn.unsqueeze(other, -1)
        d = len(cores)
        D = len(other.shape)
        for i in range(d):
            result = tn.tensordot(result, cores[i], ([D - d, -1], [2, 0]))
        return tn.squeeze(result, -1)

    return _dense_matvec_impl(other, *cores)

def bilinear_form_aux(x_cores, A_cores, y_cores, d=None):
    """
    Computes the bilinear form xT A y given the TT cores.

    Args:
        x_cores (list[torch.tensor]): the TT cores.
        A_cores (list[torch.tensor]): the TT cores.
        y_cores (list[torch.tensor]): the TT cores.
        d (int): number of modes.

    Returns:
        torch.tensor: the result as 1 element torch.tensor.
    """
    if d is None:
        d = len(A_cores)
    result = tn.ones((1, 1, 1), dtype=A_cores[0].dtype, device=A_cores[0].device)

    for i in range(d):
        result = tn.einsum('lsr,lmL->srmL',result,tn.conj(x_cores[i]))
        result = tn.einsum('srmL,smnS->LSrn',result,A_cores[i])
        result = tn.einsum('LSrn,rnR->LSR',result,y_cores[i])

    return result

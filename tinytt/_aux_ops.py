"""
Additional operations.

@author: ion
"""
import tinytt._backend as tn


def apply_mask(cores, R, indices):
    """
    compute the entries 

    Args:
        cores ([type]): [description]
        R ([type]): [description]
        indices ([type]): [description]
    """
    d = len(cores)
    dt = cores[0].dtype
    M = len(indices)
    # result = tn.zeros((M), dtype = dt)

    # for i in range(M):
    #     tmp = tn.ones(1)
    #     for k in range(d):
    #         tmp = tn.einsum('i,ik->k',tmp,cores[k][:,indices[i][k],:])
    #     result[i] = tn.sum(tmp)

    result = tn.ones((M, 1), dtype=dt, device=cores[0].device)
    for i in range(d):
        # Convert index column to Python list (tinygrad needs list for integer indexing)
        if tn.is_tensor(indices):
            idx_col = [int(tn.to_numpy(indices[j, i]).item()) for j in range(len(indices))]
        else:
            idx_col = [int(indices[j, i]) for j in range(len(indices))]
        result = tn.einsum('ij,jik->ik', result, cores[i][:, idx_col, :])

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

    fn = tn.maybe_jit(("dense_matvec", len(cores)), _dense_matvec_impl)
    return fn(other, *cores)

def bilinear_form_aux(x_cores, A_cores, y_cores, d):
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
    result = tn.ones((1,1,1), dtype = A_cores[0].dtype, device = A_cores[0].device)
    
    for i in range(d):
        result = tn.einsum('lsr,lmL->srmL',result,tn.conj(x_cores[i])) 
        result = tn.einsum('srmL,smnS->LSrn',result,A_cores[i]) 
        result = tn.einsum('LSrn,rnR->LSR',result,y_cores[i]) 
        
    return result


def tt_matvec(A_cores, x_cores, rmax=64):
    """TT-matrix @ TT-vector via per-core contraction.
    
    Contracts each TT-matrix core with the corresponding TT-vector core
    over the shared physical dimension, avoiding .full() materialisation.
    
    The output ranks multiply: r_out_k = r_A_k × r_x_k.  After all cores
    are assembled the result is rounded to ``rmax`` to control rank growth.
    
    Complexity: O(d · r_A² · r_x² · n) vs O(N · r_A²) for dense_matvec.
    For N > r_A² · r_x² this is asymptotically faster.
    
    Parameters
    ----------
    A_cores : list of Tensor
        TT-matrix cores, each (r_l_A, n_in, n_out, r_r_A).
    x_cores : list of Tensor
        TT-vector cores, each (r_l_x, n_in, r_r_x).
    rmax : int
        Max TT rank after rounding.
    
    Returns
    -------
    tt.TT
        TT-vector result of the matrix-vector product.
    """
    import tinytt as tt
    d = len(A_cores)
    new_cores = []
    for k in range(d):
        # A_core: (r_l_A, n_in, n_out, r_r_A)
        # x_core: (r_l_x, n_in, r_r_x)
        # Contraction over n_in → (r_l_A, r_l_x, n_out, r_r_A, r_r_x)
        c = tn.einsum('ainb,lim->alnbm', A_cores[k], x_cores[k])
        rl = c.shape[0] * c.shape[1]
        n = c.shape[2]
        rr = c.shape[3] * c.shape[4]
        new_cores.append(tn.reshape(c, (rl, n, rr)))
    result = tt.TT(new_cores)
    return result.round(eps=1e-10, rmax=rmax)

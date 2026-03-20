"""
Functional Tensor Train: TT wrapped around feature basis functions.
"""

from __future__ import annotations

import tinytt._backend as tn


class FunctionalTT:
    """
    Represents a function f(x_1, ..., x_d) via a TT with feature bases.

    f(x) = contract_over_d(core[d], basis[d](x[d]))

    The TT cores have shape (r_i, num_features_i, r_{i+1}) where each mode
    is contracted with the corresponding basis function outputs. The first TT
    rank encodes the output dimension. For vector-valued differential operators,
    the trailing TT rank is currently required to be 1.
    """

    def __init__(self, cores, bases):
        """
        Parameters
        ----------
        cores : list of TT-cores (3D tensors, shape (r_i, M_i, r_{i+1}))
            TT cores in the feature space. M_i = bases[i].num_features.
        bases : list of Basis objects (one per dimension)
        """
        if len(cores) != len(bases):
            raise ValueError("The number of cores must match the number of bases.")
        if len(cores) == 0:
            raise ValueError("FunctionalTT requires at least one core.")

        self._cores = cores
        self._bases = bases
        self._d = len(bases)

        for i, (core, basis) in enumerate(zip(self._cores, self._bases)):
            if core.ndim != 3:
                raise ValueError("FunctionalTT cores must be 3D tensors.")
            if core.shape[1] != basis.num_features:
                raise ValueError("Core feature dimension does not match the basis.")
            if i > 0 and self._cores[i - 1].shape[2] != core.shape[0]:
                raise ValueError("Adjacent FunctionalTT core ranks do not match.")

    @property
    def cores(self):
        return self._cores

    @property
    def bases(self):
        return self._bases

    @property
    def d(self):
        return self._d

    @property
    def output_dim(self):
        """Number of output channels represented by the first TT rank."""
        return self._cores[0].shape[0]

    @property
    def num_features(self):
        """List of feature counts per dimension."""
        return [b.num_features for b in self._bases]

    @property
    def R(self):
        """TT ranks."""
        return [c.shape[0] for c in self._cores] + [self._cores[-1].shape[2]]

    def _prepare_input(self, x):
        if x.ndim == 0:
            if self._d != 1:
                raise ValueError("Scalar input is only valid for one-dimensional FunctionalTTs.")
            return x.reshape(1, 1), True
        if x.ndim == 1:
            if self._d == 1:
                return x.unsqueeze(1), False
            return x.unsqueeze(0), True
        if x.ndim != 2 or x.shape[1] != self._d:
            raise ValueError("Expected input shape (batch, d) matching the number of bases.")
        return x, False

    def _core_eval(self, feature_map, core):
        """Contract one basis evaluation with one core, returning (batch, r_i, r_{i+1})."""
        return tn.einsum('bm,rmp->brp', feature_map, core)

    def _feature_maps(self, x, mode='value', axis=None):
        feature_maps = []
        for i, basis in enumerate(self._bases):
            if mode == 'grad' and i == axis:
                feature_maps.append(basis.grad(x[:, i]))
            elif mode == 'laplace' and i == axis:
                feature_maps.append(basis.laplace(x[:, i]))
            else:
                feature_maps.append(basis(x[:, i]))
        return feature_maps

    def _contract_feature_maps(self, feature_maps):
        state = self._core_eval(feature_maps[0], self._cores[0])
        for feature_map, core in zip(feature_maps[1:], self._cores[1:]):
            core_eval = self._core_eval(feature_map, core)
            state = tn.einsum('bij,bjk->bik', state, core_eval)
        return state

    def _finalize_output(self, state, squeeze_batch):
        out = state
        if out.shape[1] == 1:
            out = out[:, 0, :]
        if out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]
        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out.squeeze(0) if squeeze_batch else out

    def _vector_output(self, state):
        if state.shape[2] != 1:
            raise NotImplementedError("Vector-valued differential operators require the trailing TT rank to be 1.")
        return state[:, :, 0]

    def _scalar_output(self, state):
        if state.shape[1] != 1 or state.shape[2] != 1:
            raise NotImplementedError("grad currently requires scalar FunctionalTT outputs.")
        return state[:, 0, 0]

    def __call__(self, x):
        """Evaluate f at points x."""
        x, squeeze_batch = self._prepare_input(x)
        feature_maps = self._feature_maps(x)
        state = self._contract_feature_maps(feature_maps)
        return self._finalize_output(state, squeeze_batch)

    def grad(self, x, eps=1e-7):
        """Compute the gradient of a scalar FunctionalTT."""
        _ = eps
        x, squeeze_batch = self._prepare_input(x)

        grads = []
        for j in range(x.shape[1]):
            state = self._contract_feature_maps(self._feature_maps(x, mode='grad', axis=j))
            grads.append(self._scalar_output(state))

        grad = tn.stack(grads, dim=1)
        return grad.squeeze(0) if squeeze_batch else grad

    def jacobian(self, x):
        """Compute the Jacobian with shape `(batch, output_dim, d)` or `(batch, d)` for scalar outputs."""
        x, squeeze_batch = self._prepare_input(x)

        cols = []
        for j in range(x.shape[1]):
            state = self._contract_feature_maps(self._feature_maps(x, mode='grad', axis=j))
            cols.append(self._vector_output(state))

        jac = tn.stack(cols, dim=2)
        if self.output_dim == 1:
            jac = jac[:, 0, :]
        return jac.squeeze(0) if squeeze_batch else jac

    def laplace(self, x):
        """Compute the Laplacian with shape `(batch, output_dim)` or `(batch,)` for scalar outputs."""
        x, squeeze_batch = self._prepare_input(x)

        laps = []
        for k in range(self._d):
            state = self._contract_feature_maps(self._feature_maps(x, mode='laplace', axis=k))
            laps.append(self._vector_output(state))

        lap = tn.stack(laps, dim=2).sum(axis=2)
        if self.output_dim == 1:
            lap = lap[:, 0]
        return lap.squeeze(0) if squeeze_batch else lap

    def divergence(self, x, upto=None):
        """Compute divergence by summing `dF_i/dx_i` over the first `upto` channels."""
        x, squeeze_batch = self._prepare_input(x)
        upto = self._d if upto is None else upto
        if upto < 0 or upto > self._d:
            raise ValueError("upto must satisfy 0 <= upto <= d.")
        if self.output_dim < upto:
            raise ValueError("divergence requires output_dim >= upto.")

        div = tn.zeros((x.shape[0],), dtype=self._cores[0].dtype, device=self._cores[0].device)
        for mu in range(upto):
            state = self._contract_feature_maps(self._feature_maps(x, mode='grad', axis=mu))
            div = div + self._vector_output(state)[:, mu]

        return div.squeeze(0) if squeeze_batch else div

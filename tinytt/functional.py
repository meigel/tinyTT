"""
Functional Tensor Train: a small experimental TT layer for basis-driven models.

This module intentionally exposes a narrow subset: one univariate basis per
input dimension and explicit scalar/vector operator APIs built on top of one
shared contraction path.
"""

from __future__ import annotations

import tinytt._backend as tn


class FunctionalTT:
    """
    Represents a function f(x_1, ..., x_d) via a TT with feature bases.

    f(x) = contract_over_d(core[d], basis[d](x[d]))

    This is the supported functional-TT subset in tinyTT: TT cores have shape
    (r_i, num_features_i, r_{i+1}). Vector outputs are supported when exactly
    one TT boundary rank is nontrivial, so the output layout stays unambiguous.
    """

    def __init__(self, cores, bases):
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
    def output_layout(self):
        left_rank = self._cores[0].shape[0]
        right_rank = self._cores[-1].shape[2]
        if left_rank == 1 and right_rank == 1:
            return 'scalar'
        if right_rank == 1:
            return 'leading'
        if left_rank == 1:
            return 'trailing'
        return 'ambiguous'

    @property
    def output_dim(self):
        layout = self.output_layout
        if layout == 'scalar':
            return 1
        if layout == 'leading':
            return self._cores[0].shape[0]
        if layout == 'trailing':
            return self._cores[-1].shape[2]
        raise NotImplementedError('output_dim is ambiguous when both TT boundary ranks are nontrivial.')

    @property
    def num_features(self):
        return [b.num_features for b in self._bases]

    @property
    def R(self):
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
        return tn.einsum('bm,rmp->brp', feature_map, core)

    def _build_feature_maps(self, x, orders=None):
        orders = [0] * self._d if orders is None else list(orders)
        feature_maps = []
        for i, basis in enumerate(self._bases):
            order = orders[i]
            if order == 0:
                feature_maps.append(basis(x[:, i]))
            elif order == 1:
                feature_maps.append(basis.grad(x[:, i]))
            elif order == 2:
                feature_maps.append(basis.laplace(x[:, i]))
            else:
                raise ValueError("Only derivative orders 0, 1, and 2 are supported.")
        return feature_maps

    def _contract_feature_maps(self, feature_maps):
        state = self._core_eval(feature_maps[0], self._cores[0])
        for feature_map, core in zip(feature_maps[1:], self._cores[1:]):
            state = tn.einsum('bij,bjk->bik', state, self._core_eval(feature_map, core))
        return state

    def _evaluate_state(self, x, orders=None):
        return self._contract_feature_maps(self._build_feature_maps(x, orders=orders))

    def _format_vector_state(self, state):
        layout = self.output_layout
        if layout == 'scalar':
            return state[:, 0, :]
        if layout == 'leading':
            if state.shape[2] != 1:
                raise NotImplementedError('Leading-layout vector outputs require the trailing TT rank to be 1.')
            return state[:, :, 0]
        if layout == 'trailing':
            if state.shape[1] != 1:
                raise NotImplementedError('Trailing-layout vector outputs require the leading TT rank to be 1.')
            return state[:, 0, :]
        raise NotImplementedError('Vector outputs are ambiguous when both TT boundary ranks are nontrivial.')

    def _format_value_output(self, state):
        vector = self._format_vector_state(state)
        if vector.ndim == 2 and vector.shape[1] == 1:
            return vector[:, 0]
        return vector

    def _scalar_output(self, state):
        vector = self._format_vector_state(state)
        if vector.shape[1] != 1:
            raise NotImplementedError('grad currently requires scalar FunctionalTT outputs.')
        return vector[:, 0]

    def _apply_operator(self, x, orders=None, output='value'):
        state = self._evaluate_state(x, orders=orders)
        if output == 'value':
            return self._format_value_output(state)
        if output == 'vector':
            return self._format_vector_state(state)
        if output == 'scalar':
            return self._scalar_output(state)
        raise ValueError(f'Unsupported output mode: {output}')

    def __call__(self, x):
        x, squeeze_batch = self._prepare_input(x)
        out = self._apply_operator(x, output='value')
        return out.squeeze(0) if squeeze_batch else out

    def grad(self, x, eps=1e-7):
        _ = eps
        x, squeeze_batch = self._prepare_input(x)
        cols = [self._apply_operator(x, orders=[1 if j == axis else 0 for j in range(self._d)], output='scalar') for axis in range(self._d)]
        grad = tn.stack(cols, dim=1)
        return grad.squeeze(0) if squeeze_batch else grad

    def jacobian(self, x):
        x, squeeze_batch = self._prepare_input(x)
        cols = [self._apply_operator(x, orders=[1 if j == axis else 0 for j in range(self._d)], output='vector') for axis in range(self._d)]
        jac = tn.stack(cols, dim=2)
        if self.output_dim == 1:
            jac = jac[:, 0, :]
        return jac.squeeze(0) if squeeze_batch else jac

    def laplace(self, x):
        x, squeeze_batch = self._prepare_input(x)
        cols = [self._apply_operator(x, orders=[2 if j == axis else 0 for j in range(self._d)], output='vector') for axis in range(self._d)]
        lap = tn.stack(cols, dim=2).sum(axis=2)
        if self.output_dim == 1:
            lap = lap[:, 0]
        return lap.squeeze(0) if squeeze_batch else lap

    def divergence(self, x, upto=None):
        x, squeeze_batch = self._prepare_input(x)
        upto = self._d if upto is None else upto
        if upto < 0 or upto > self._d:
            raise ValueError('upto must satisfy 0 <= upto <= d.')
        jac = self.jacobian(x)
        if self.output_dim == 1:
            if upto > 1:
                raise ValueError('divergence requires output_dim >= upto.')
            div = jac[:, 0] if jac.ndim == 2 else jac[0]
            return div.squeeze(0) if squeeze_batch and getattr(div, 'ndim', 0) > 0 else div
        if jac.shape[1] < upto:
            raise ValueError('divergence requires output_dim >= upto.')
        div = tn.zeros((x.shape[0],), dtype=self._cores[0].dtype, device=self._cores[0].device)
        for mu in range(upto):
            div = div + jac[:, mu, mu]
        return div.squeeze(0) if squeeze_batch else div

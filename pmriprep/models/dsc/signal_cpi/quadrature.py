"""Checked gamma quadrature retaining unchanged interval contributions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaincinv, gammaln, roots_jacobi

from .batch import EnsembleConvolution
from .dispersion import DispersionIntegrationError
from .prepared import PreparedCPIConvolution


class IntervalReuseConvolution(EnsembleConvolution):
    """Retain coarse/fine contributions until their interval is actually split.

    Notes
    -----
    All complete sums, per-row error tolerances and refinement decisions are
    recalculated each round. Integral reuse is local to one call; no changed
    parameter vector may inherit another call's interval contributions.
    """

    def _gamma(self, knots: np.ndarray, values: np.ndarray, target: np.ndarray,
               gamma: np.ndarray) -> np.ndarray:
        """Evaluate the same adaptive quadrature using only new child intervals."""
        self.last_rounds: list[dict[str, Any]] = []
        if not self.base.left.size:
            return np.zeros_like(target)
        prepared = PreparedCPIConvolution.from_operator(self.base)
        shape = 1 + gamma[:, 0] * gamma[:, 1]
        rate = gamma[:, 1]
        probabilities = np.array(
            [1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-12]
        )
        transport = gammaincinv(shape[:, None], probabilities) / rate[:, None]
        parts = []
        for row in range(target.shape[0]):
            if np.max(target[row]) <= self.base.left[0]:
                continue
            prepared._prepare(knots[row], values[row], float(np.max(target[row])))
            edges = prepared._breaks
            cuts = np.sort(
                np.clip(
                    np.column_stack(
                        (target[row, :, None] - edges[None, :],
                         np.broadcast_to(transport[row], (target.shape[1], probabilities.size)))
                    ), 0, np.maximum(target[row] - self.base.left[0], 0)[:, None]
                ), axis=1
            )
            lower, upper = cuts[:, :-1], cuts[:, 1:]
            frames, columns = np.nonzero(upper > lower)
            lo, hi = lower[frames, columns], upper[frames, columns]
            midpoint = target[row, frames] - (lo + hi) / 2
            indices = np.clip(
                np.searchsorted(edges, midpoint, side="right") - 1, 0, edges.size - 2
            )
            parts.append((np.full(frames.size, row), frames, lo, hi,
                          edges[indices], edges[indices + 1], prepared._coefficients[:, indices].T))
        if not parts:
            return np.zeros_like(target)
        rows, frames, lo, hi, poly_lo, poly_hi, coefficients = (
            np.concatenate([part[i] for part in parts]) for i in range(7)
        )
        jacobi = {}
        unique_shapes, shape_index = np.unique(shape, return_inverse=True)
        for order in self.rules:
            nodes, weights = [], []
            for value in unique_shapes:
                x, w = roots_jacobi(order, 0, value - 1)
                nodes.append(x)
                weights.append(w / np.sum(w))
            jacobi[order] = (np.array(nodes), np.array(weights))

        def integrate(order: int, first: int) -> np.ndarray:
            """Evaluate the unevaluated suffix in the current interval ordering."""
            nodes, weights = self.rules[order]
            row_ids, frame_ids = rows[first:], frames[first:]
            left, right = lo[first:], hi[first:]
            width = right - left
            lag = left[:, None] + width[:, None] * (nodes + 1) / 2
            sh, ra = shape[row_ids, None], rate[row_ids, None]
            log_weights = (
                np.log(width[:, None] * weights / 2) + sh * np.log(ra)
                - gammaln(sh) + (sh - 1) * np.log(lag) - ra * lag
            )
            origin = left == 0
            if np.any(origin):
                jnodes, jweights = jacobi[order]
                ids = shape_index[row_ids[origin]]
                lag[origin] = right[origin, None] * (jnodes[ids] + 1) / 2
                log_weights[origin] = (
                    np.log(jweights[ids]) + sh[origin] * np.log(ra[origin] * right[origin, None])
                    - gammaln(sh[origin] + 1) - ra[origin] * lag[origin]
                )
            local = (
                2 * (target[row_ids, frame_ids, None] - lag - poly_lo[first:, None])
                / (poly_hi - poly_lo)[first:, None] - 1
            )
            response = np.polynomial.chebyshev.chebval(
                local, coefficients[first:].T[:, :, None], tensor=False
            )
            return np.asarray(np.sum(response * np.exp(log_weights), axis=1))

        first = 0
        coarse, fine = np.empty(rows.size), np.empty(rows.size)
        for iteration in range(30):
            coarse[first:], fine[first:] = integrate(16, first), integrate(32, first)
            error = 4 * np.abs(fine - coarse)
            ids = rows * target.shape[1] + frames
            total = np.bincount(ids, weights=fine, minlength=target.size).reshape(target.shape)
            errors = np.bincount(ids, weights=error, minlength=target.size).reshape(target.shape)
            tolerance = np.maximum(1e-200, 2e-12 * np.max(np.abs(total), axis=1) / 8)
            self.last_rounds.append(dict(iteration=iteration,
                intervals=np.bincount(rows, minlength=len(target)).tolist(),
                evaluated=rows.size-first,
                max_error_ratio=float(np.max(errors/tolerance[:, None]))))
            if np.all(np.isfinite(total)) and np.all(errors <= tolerance[:, None]):
                return np.asarray(total, dtype=float)
            count = np.bincount(ids, minlength=target.size)
            refine = error > tolerance[rows] / np.maximum(1, count[ids])
            sizes = np.bincount(rows, minlength=target.shape[0]) + np.bincount(
                rows[refine], minlength=target.shape[0]
            )
            if not np.any(refine) or np.any(sizes > 100000):
                break
            keep = ~refine
            first = int(np.count_nonzero(keep))
            children = 2 * int(np.count_nonzero(refine))
            coarse = np.r_[coarse[keep], np.empty(children)]
            fine = np.r_[fine[keep], np.empty(children)]
            middle = (lo[refine] + hi[refine]) / 2
            lo, hi = (
                np.r_[lo[keep], lo[refine], middle],
                np.r_[hi[keep], middle, hi[refine]],
            )
            rows, frames, poly_lo, poly_hi, coefficients = (
                np.concatenate((x[keep], x[refine], x[refine]))
                for x in (rows, frames, poly_lo, poly_hi, coefficients)
            )
        raise DispersionIntegrationError("ensemble gamma quadrature did not meet integration accuracy")

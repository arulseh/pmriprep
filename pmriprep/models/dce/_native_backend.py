"""Validated memory ownership and SciPy ABI bindings for native DCE fitting."""

from __future__ import annotations

import ctypes as ct
import importlib
from threading import Lock
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.linalg import cython_lapack
from scipy.special import cython_special
from scipy.optimize import OptimizeResult

from pmriprep.models.dce.fit_errors import NumericalFitError
from pmriprep.utils.dce_constants import (
    EPSILON_FLOW,
    EPSILON_VE_FORWARD,
    EPSILON_VP,
    EPSILON_EIGENVALUE,
)

if TYPE_CHECKING:
    from pmriprep.models.dce.native_exchange import NativeInput

try:
    _kernel = importlib.import_module("pmriprep.models.dce._native_trf")
except ImportError as exc:
    raise ImportError(
        "Native DCE fitting requires the built PMRIPrep extension."
    ) from exc

D = ct.POINTER(ct.c_double)
INT_POINTER = ct.POINTER(ct.c_int64)


class _Context(ct.Structure):
    """C-compatible context retaining separate private work arrays."""

    _fields_ = (
        [(k, ct.c_size_t) for k in ["frames", "observations", "unique"]]
        + [
            (k, ct.c_double)
            for k in ["h", "baseline", "eps_flow", "eps_ve", "eps_vp", "eps_eigen"]
        ]
        + [(k, D) for k in ["full", "part", "s", "q", "held", "tail", "unique_s"]]
        + [(k, INT_POINTER) for k in ["k", "inverse"]]
        + [(k, D) for k in ["moments", "states", "high_curve", "low_curve", "scratch"]]
        + [("hyper", ct.c_void_p)]
    )


EVALUATE = ct.CFUNCTYPE(
    None, ct.POINTER(_Context), D, D, ct.c_double, ct.c_int, D, D, D
)(_kernel.evaluate)
RECURRENCE = ct.CFUNCTYPE(None, D, D, D, ct.c_size_t, ct.c_size_t)(_kernel.recurrence)
SVD = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_int, D, D, D, D)(_kernel.svd)
STEP = ct.CFUNCTYPE(
    ct.c_int, ct.c_int, D, D, D, D, D, D, D, ct.c_double, D, D, ct.c_double, D
)(_kernel.step)
SOLVE = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    D,
    ct.c_double,
    D,
    ct.c_void_p,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_int,
    D,
    D,
    D,
)(_kernel.solve)


def special_pointer() -> tuple[int, str]:
    """Require the exact installed real hyp1f1 capsule signature."""
    capsule = cython_special.__pyx_capi__.get("__pyx_fuse_1hyp1f1")
    if capsule is None:
        raise RuntimeError("Required SciPy hyp1f1 capsule is missing")
    get_name = ct.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = [ct.py_object], ct.c_char_p
    name = get_name(capsule)
    if name != b"double (double, double, double, int __pyx_skip_dispatch)":
        raise RuntimeError("Unsupported SciPy hyp1f1 capsule signature")
    get_pointer = ct.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes, get_pointer.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
    return int(get_pointer(capsule, name)), name.decode()


class Evaluator:
    """Own residual/Jacobian workspace and return independent outputs.

    Parameters
    ----------
    context : NativeInput
        Prepared nonnegative-offset plasma input and observation support.
    curve : ndarray
        Finite tissue concentration in mM at retained observations.

    Notes
    -----
    Native calls on this evaluator are serialized. Input arrays are copied
    so later changes to the input context do not invalidate native pointers.
    """

    def __init__(self, context: NativeInput, curve: np.ndarray) -> None:
        self.curve = np.ascontiguousarray(curve, dtype=np.float64).copy()
        self.n = len(context.output_t)
        if not 1 <= self.n <= 100000:
            raise ValueError("Native fitting requires 1 to 100000 observations.")
        if self.curve.shape != (self.n,) or not np.isfinite(self.curve).all():
            raise ValueError("Finite curve matching observation support required.")
        self.scale = max(float(np.max(abs(self.curve))), 1e-5)
        self.function = EVALUATE
        self._lock = Lock()
        self.arrays: dict[str, np.ndarray] = {}
        c = _Context()
        c.frames, c.observations, c.unique = (
            len(context.t),
            self.n,
            len(context.unique_s),
        )
        c.h, c.baseline = context.h, float(context.aif[0])
        if (
            c.frames < 2
            or c.unique < 1
            or not np.isfinite(c.h)
            or c.h <= 0
            or not np.isfinite(c.baseline)
        ):
            raise ValueError("Finite nonempty native input geometry required.")
        c.eps_flow, c.eps_ve, c.eps_vp, c.eps_eigen = (
            EPSILON_FLOW,
            EPSILON_VE_FORWARD,
            EPSILON_VP,
            EPSILON_EIGENVALUE,
        )
        for name, attr in [
            ("full", "full_coeff"),
            ("part", "part_coeff"),
            ("s", "s"),
            ("q", "q"),
            ("held", "held"),
            ("tail", "tail"),
            ("unique_s", "unique_s"),
        ]:
            a = np.ascontiguousarray(getattr(context, attr), dtype=np.float64).copy()
            expected = {
                "full": (4, c.frames - 1),
                "part": (4, self.n),
                "unique_s": (c.unique,),
            }.get(name, (self.n,))
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError(f"Invalid native input array {name}.")
            self.arrays[name] = a
            setattr(c, name, a.ctypes.data_as(D))
        for name, attr in [("k", "k"), ("inverse", "inverse_s")]:
            indices = np.ascontiguousarray(
                getattr(context, attr), dtype=np.int64
            ).copy()
            limit = c.frames - 1 if name == "k" else c.unique
            if (
                indices.shape != (self.n,)
                or np.any(indices < 0)
                or np.any(indices >= limit)
            ):
                raise ValueError(f"Invalid native input indices {name}.")
            self.arrays[name] = indices
            setattr(c, name, indices.ctypes.data_as(INT_POINTER))
        for name, shape in [
            ("moments", (4, c.unique + 1)),
            ("states", (c.frames,)),
            ("high_curve", (self.n,)),
            ("low_curve", (self.n,)),
            ("scratch", (self.n,)),
        ]:
            a = np.empty(shape, dtype=np.float64)
            self.arrays[name] = a
            setattr(c, name, a.ctypes.data_as(D))
        c.hyper, self.signature = special_pointer()
        self.context = c
        self.context_pointer = ct.pointer(c)
        self.curve_pointer = self.curve.ctypes.data_as(D)

    def evaluate(
        self, x: np.ndarray, jacobian: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prediction, scaled residual and optional finite differences.

        Parameters
        ----------
        x : ndarray
            Four finite bounded optimizer coordinates.
        jacobian : bool
            Evaluate base plus four unchanged finite-difference perturbations.

        Returns
        -------
        tuple of ndarray
            Independent prediction, residual and frame-by-parameter Jacobian.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        if (
            x.shape != (4,)
            or not np.isfinite(x).all()
            or np.any(x < 0)
            or np.any(x > [5.0, 5.0, 1.0, 1.0])
        ):
            raise ValueError("Four finite bounded optimizer coordinates required.")
        predicted, residual = np.empty(self.n), np.empty(self.n)
        jac = np.empty((self.n, 4) if jacobian else (0, 4))
        with self._lock:
            self.function(
                self.context_pointer,
                x.ctypes.data_as(D),
                self.curve_pointer,
                self.scale,
                int(jacobian),
                predicted.ctypes.data_as(D),
                residual.ctypes.data_as(D),
                jac.ctypes.data_as(D),
            )
        return predicted, residual, jac

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Return a fresh scaled residual vector."""
        return self.evaluate(x)[1]

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return a fresh numerical Jacobian matrix."""
        return self.evaluate(x, True)[2]


def lapack_pointer() -> tuple[int, str]:
    """Require the installed double-precision, LP64 SVD capsule signature."""
    cap = cython_lapack.__pyx_capi__.get("dgesdd")
    if cap is None:
        raise RuntimeError("Required SciPy LAPACK SVD capsule is missing")
    name_fn = ct.pythonapi.PyCapsule_GetName
    name_fn.argtypes, name_fn.restype = [ct.py_object], ct.c_char_p
    name = name_fn(cap)
    double = "__pyx_t_5scipy_6linalg_13cython_lapack_d *"
    expected = (
        "void ("
        + ", ".join(
            [
                "char *",
                "int *",
                "int *",
                double,
                "int *",
                double,
                double,
                "int *",
                double,
                "int *",
                double,
                "int *",
                "int *",
                "int *",
            ]
        )
        + ")"
    )
    if name.decode() != expected:
        raise RuntimeError("Unexpected SciPy LAPACK SVD signature")
    ptr_fn = ct.pythonapi.PyCapsule_GetPointer
    ptr_fn.argtypes, ptr_fn.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
    return int(ptr_fn(cap, name)), name.decode()


LAPACK, SIGNATURE = lapack_pointer()


def check_backend() -> None:
    """Check compiled SciPy interfaces before DCE processing starts.

    Raises
    ------
    RuntimeError
        If the installed SciPy special-function or LAPACK interface is
        unavailable or has an incompatible signature. Importing this module
        also requires the packaged native extension.
    """
    lapack_pointer()
    special_pointer()


def pointer(value: np.ndarray) -> Any:
    """Return a pointer while the caller retains the owning array."""
    return value.ctypes.data_as(D)


def svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the installed gesdd backend with explicit array layouts."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 4
        or matrix.shape[0] < 4
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Finite tall four-column matrix required")
    u, s, vt = (
        np.empty((matrix.shape[0], 4), order="F"),
        np.empty(4),
        np.empty((4, 4), order="F"),
    )
    code = SVD(
        LAPACK, len(matrix), pointer(matrix), pointer(u), pointer(s), pointer(vt)
    )
    if code:
        raise RuntimeError(f"Native SVD error {code}")
    return u, s, vt


def step(
    m: int,
    x: np.ndarray,
    J: np.ndarray,
    diag: np.ndarray,
    g: np.ndarray,
    p: np.ndarray,
    ph: np.ndarray,
    d: np.ndarray,
    radius: float,
    lb: np.ndarray,
    ub: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray, np.ndarray]:
    """Evaluate step selection, returning isolated mutated direction copies."""
    values = [
        np.ascontiguousarray(v, dtype=np.float64).copy()
        for v in [x, J, diag, g, p, ph, d, lb, ub]
    ]
    if (
        m < 1
        or values[1].shape != (m, 4)
        or any(v.shape != (4,) for i, v in enumerate(values) if i != 1)
    ):
        raise ValueError("Four-coordinate vectors and frame-by-four Jacobian required")
    if (
        not all(np.isfinite(v).all() for v in values)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("Finite step state required")
    out = np.empty(9)
    code = STEP(
        m,
        *[pointer(v) for v in values[:7]],
        radius,
        pointer(values[7]),
        pointer(values[8]),
        theta,
        pointer(out),
    )
    if code < 0:
        raise RuntimeError(f"Native step error {code}")
    return out[:4], out[4:8], float(out[8]), code, values[4], values[5]


def solve(
    evaluator: Evaluator,
    x0: np.ndarray,
    max_nfev: int = 180,
    ftol: float = 2e-9,
    xtol: float = 2e-9,
    gtol: float = 2e-9,
    trace: bool = False,
) -> OptimizeResult:
    """Solve one fixed four-coordinate problem with private native workspace.

    Parameters
    ----------
    evaluator : Evaluator
        Existing compiled PCHIP residual and numerical-Jacobian context.
    x0 : ndarray
        Four finite coordinates inside the unchanged box bounds.
    max_nfev : int
        Unchanged residual evaluation budget.
    ftol, xtol, gtol : float
        Positive finite stopping tolerances.
    trace : bool
        Return iteration states for diagnostic comparison.

    Returns
    -------
    OptimizeResult
        Coordinates, status, counters, objective and optional iteration trace.

    Raises
    ------
    NumericalFitError
        Known numerical failure in this fit, including nonfinite solver output.
    MemoryError
        Native workspace allocation failed.
    RuntimeError
        Unclassified solver, LAPACK or backend failure; must not be treated as
        an unavailable voxel.
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if (
        x0.shape != (4,)
        or not np.isfinite(x0).all()
        or np.any(x0 < 0)
        or np.any(x0 > [5, 5, 1, 1])
    ):
        raise ValueError("Four finite bounded initial coordinates required")
    if not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1 or max_nfev > 100000:
        raise ValueError("Positive bounded integer evaluation budget required")
    if not all(np.isfinite(v) and v > 0 for v in [ftol, xtol, gtol]):
        raise ValueError("Positive finite tolerances required")
    result, info = np.empty(4), np.empty(7)
    record = np.zeros((max_nfev + 2, 16)) if trace else np.empty((0, 16))
    with evaluator._lock:
        code = SOLVE(
            evaluator.context_pointer,
            evaluator.curve_pointer,
            evaluator.scale,
            pointer(x0),
            LAPACK,
            ftol,
            xtol,
            gtol,
            max_nfev,
            pointer(result),
            pointer(info),
            pointer(record) if trace else None,
        )
    if code == -11:
        raise MemoryError("Native TRF allocation failed")
    if code in (-13, -14, -15):
        raise NumericalFitError(f"Native TRF numerical error {code}")
    if code:
        raise RuntimeError(f"Native TRF numerical error {code}")
    if not np.isfinite(result).all() or not np.isfinite(info).all():
        raise NumericalFitError("Nonfinite native result")
    return OptimizeResult(
        x=result,
        status=int(info[0]),
        nfev=int(info[1]),
        njev=int(info[2]),
        nit=int(info[3]),
        cost=float(info[4]),
        optimality=float(info[5]),
        trace=record[: int(info[6])] if trace else None,
    )

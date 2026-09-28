#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static Py_ssize_t bound(const double *x, Py_ssize_t n, double v, int right) {
    Py_ssize_t lo = 0, hi = n;
    while (lo < hi) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (x[mid] < v || (right && x[mid] == v)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static double polynomial(const double *c, Py_ssize_t stride, Py_ssize_t degree,
                         double x, int chebyshev) {
    if (!chebyshev) {
        double value = 0.;
        for (Py_ssize_t j = 0; j <= degree; ++j) value = value*x + c[j*stride];
        return value;
    }
    if (degree == 0) return c[0];
    double c0 = c[(degree-1)*stride], c1 = c[degree*stride];
    for (Py_ssize_t j = degree-2; j >= 0; --j) {
        double previous = c0;
        c0 = c[j*stride] - c1;
        c1 = previous + c1*(2*x);
    }
    return c0 + c1*x;
}

static void integrate(Py_ssize_t nr, Py_ssize_t nf, Py_ssize_t na, Py_ssize_t nk,
                      Py_ssize_t nq, Py_ssize_t degree, int chebyshev,
                      const double *knots, const double *target, const double *residue,
                      const double *left, const double *right, const double *arterial,
                      const double *nodes, const double *weights, const double *moments,
                      double *out) {
    for (Py_ssize_t row = 0; row < nr; ++row) {
        const double *k = knots + row*nk;
        const double *coef = residue + row*4*nk;
        for (Py_ssize_t frame = 0; frame < nf; ++frame) {
            double t = target[row*nf+frame], total = 0.;
            for (Py_ssize_t a = 0; a < na; ++a) {
                double rawlo = t-right[a], rawhi = t-left[a];
                double lo = fmax(rawlo, 0.), hi = rawhi;
                if (!(hi > lo)) continue;
                Py_ssize_t first = bound(k, nk, lo, 1)-1;
                Py_ssize_t last = bound(k, nk, hi, 0)-1;
                for (Py_ssize_t r = first; r <= last; ++r) {
                    double lower = fmax(lo, k[r]);
                    double upper = r+1 < nk ? fmin(hi, k[r+1]) : hi;
                    double c0=coef[r], c1=coef[nk+r], c2=coef[2*nk+r], c3=coef[3*nk+r];
                    double value;
                    if (moments != NULL && lower == rawlo && upper == rawhi) {
                        double x = (upper+lower)/2-k[r], h = (right[a]-left[a])/2;
                        value = (((c0*x+c1)*x+c2)*x+c3)*moments[a]
                            + ((3*c0*x+2*c1)*x+c2)*h*moments[na+a]
                            + (3*c0*x+c1)*(h*h)*moments[2*na+a]
                            + c0*(h*h*h)*moments[3*na+a];
                    } else {
                        double half = (upper-lower)/2, middle = (upper+lower)/2;
                        double subtotal = 0.;
                        for (Py_ssize_t q = 0; q < nq; ++q) {
                            double point = middle+half*nodes[q], x = point-k[r];
                            double rv = ((c0*x+c1)*x+c2)*x+c3;
                            double alocal = t-point-left[a];
                            if (chebyshev) alocal = alocal/((right[a]-left[a])/2)-1;
                            double av = polynomial(arterial+a, na, degree, alocal, chebyshev);
                            subtotal += (av*rv)*weights[q];
                        }
                        value = half*subtotal;
                    }
                    total += value;
                }
            }
            out[row*nf+frame] = total;
        }
    }
}

static int validate_typed_buffer(Py_buffer *v, int ndim, const char *name,
                                  int indices, int any_order) {
    Py_ssize_t itemsize = indices ? sizeof(Py_ssize_t) : sizeof(double);
    const char *format = v->format;
    if (format && (*format == '@' || *format == '=')) ++format;
    int valid_format = format && (indices
        ? (strcmp(format, "n") == 0 ||
           (sizeof(long) == sizeof(Py_ssize_t) && strcmp(format, "l") == 0) ||
           (sizeof(long long) == sizeof(Py_ssize_t) && strcmp(format, "q") == 0) ||
           (sizeof(int) == sizeof(Py_ssize_t) && strcmp(format, "i") == 0))
        : strcmp(format, "d") == 0);
    if (v->ndim != ndim || v->shape == NULL || v->strides == NULL || v->format == NULL ||
        !valid_format || v->itemsize != itemsize || sizeof(double) != 8 ||
        !PyBuffer_IsContiguous(v, any_order ? 'A' : 'C') ||
        v->suboffsets != NULL || (uintptr_t)v->buf % itemsize != 0) {
        PyErr_Format(PyExc_ValueError, "%s must be aligned contiguous native %s with %d dimensions",
                     name, indices ? "intp" : "float64", ndim);
        return 0;
    }
    Py_ssize_t bytes = itemsize;
    for (int j = 0; j < ndim; ++j) {
        if (v->shape[j] < 0 || (v->shape[j] && bytes > PY_SSIZE_T_MAX / v->shape[j])) {
            PyErr_Format(PyExc_ValueError, "%s buffer shape exceeds its addressable extent", name);
            return 0;
        }
        bytes *= v->shape[j];
    }
    if (bytes != v->len || (bytes && v->buf == NULL) ||
        (uintptr_t)v->buf > UINTPTR_MAX - (uintptr_t)bytes) {
        PyErr_Format(PyExc_ValueError, "%s buffer extent disagrees with its shape", name);
        return 0;
    }
    return 1;
}

static int validate_buffer(Py_buffer *v, int ndim, const char *name) {
    return validate_typed_buffer(v, ndim, name, 0, 0);
}

static PyObject *convolve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *objects[10], *chebyshev_object;
    Py_buffer v[10] = {{0}};
    const char *names[] = {"knots", "coefficients", "target", "left", "right",
                          "arterial", "nodes", "weights", "moments", "output"};
    int ndim[] = {2, 3, 2, 1, 1, 2, 1, 1, 2, 2};
    if (!PyArg_UnpackTuple(args, "convolve", 11, 11, &objects[0], &objects[1], &objects[2],
                          &objects[3], &objects[4], &objects[5], &objects[6], &objects[7],
                          &objects[8], &objects[9], &chebyshev_object)) return NULL;
    int chebyshev = PyObject_IsTrue(chebyshev_object);
    if (chebyshev < 0) return NULL;
    for (int i = 0; i < 10; ++i) {
        if (i == 8 && objects[i] == Py_None) continue;
        int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | (i == 9 ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(objects[i], &v[i], flags) < 0) goto fail;
        if (!validate_buffer(&v[i], ndim[i], names[i])) goto fail;
    }
    Py_ssize_t nr=v[0].shape[0], nk=v[0].shape[1], nf=v[2].shape[1];
    Py_ssize_t na=v[3].shape[0], nq=v[6].shape[0], nc=v[5].shape[0];
    if (nk < 2 || nq < 1 || nc < 1 || v[1].shape[0] != nr || v[1].shape[1] != 4 ||
        v[1].shape[2] != nk || v[2].shape[0] != nr || v[4].shape[0] != na ||
        v[5].shape[1] != na || v[7].shape[0] != nq || v[9].shape[0] != nr ||
        v[9].shape[1] != nf || (v[8].obj && (v[8].shape[0] != 4 || v[8].shape[1] != na))) {
        PyErr_SetString(PyExc_ValueError, "incompatible convolution array shapes");
        goto fail;
    }
    for (int i = 0; i < 9; ++i) {
        if (!v[i].obj) continue;
        if (v[i].len && v[9].len && (uintptr_t)v[i].buf < (uintptr_t)v[9].buf + v[9].len &&
            (uintptr_t)v[9].buf < (uintptr_t)v[i].buf + v[i].len) {
            PyErr_SetString(PyExc_ValueError, "output must not overlap any input buffer");
            goto fail;
        }
        const double *x = v[i].buf;
        for (Py_ssize_t j = 0; j < v[i].len / (Py_ssize_t)sizeof(double); ++j) {
            if (!isfinite(x[j])) {
                PyErr_Format(PyExc_ValueError, "%s must contain only finite values", names[i]);
                goto fail;
            }
        }
    }
    const double *knots = v[0].buf, *left = v[3].buf, *right = v[4].buf;
    for (Py_ssize_t row = 0; row < nr; ++row) {
        if (knots[row*nk] != 0.) {
            PyErr_SetString(PyExc_ValueError, "residue knots must start at zero");
            goto fail;
        }
        for (Py_ssize_t j = 1; j < nk; ++j) {
            if (knots[row*nk+j] <= knots[row*nk+j-1]) {
                PyErr_SetString(PyExc_ValueError, "residue knots must be strictly increasing");
                goto fail;
            }
        }
    }
    for (Py_ssize_t j = 0; j < na; ++j) {
        if (right[j] <= left[j]) {
            PyErr_SetString(PyExc_ValueError, "arterial intervals must have positive width");
            goto fail;
        }
    }
    integrate(nr, nf, na, nk, nq, nc-1, chebyshev, knots, v[2].buf, v[1].buf,
              left, right, v[5].buf, v[6].buf, v[7].buf,
              v[8].obj ? v[8].buf : NULL, v[9].buf);
    for (int i = 0; i < 10; ++i) if (v[i].obj) PyBuffer_Release(&v[i]);
    Py_RETURN_NONE;
fail:
    for (int i = 0; i < 10; ++i) if (v[i].obj) PyBuffer_Release(&v[i]);
    return NULL;
}

static Py_ssize_t enumerate_intervals(Py_ssize_t nt, Py_ssize_t na,
        const double *target, const double *left, const double *right,
        double lower_cut, double upper_cut, unsigned char *packed, char **out) {
    Py_ssize_t count = 0;
    for (Py_ssize_t r = 0; r < nt; ++r) {
        for (Py_ssize_t s = 0; s < na; ++s) {
            double low = fmax(0., target[r]-right[s]);
            double high = fmin(upper_cut, fmax(0., target[r]-left[s]));
            if (!(high > low && high > lower_cut)) continue;
            if (out) {
                memcpy(out[0]+count*sizeof(r), &r, sizeof(r));
                memcpy(out[1]+count*sizeof(s), &s, sizeof(s));
                memcpy(out[2]+count*sizeof(low), &low, sizeof(low));
                memcpy(out[3]+count*sizeof(high), &high, sizeof(high));
            } else {
                Py_ssize_t bit = r*na+s;
                packed[bit/8] |= (unsigned char)(1u << (7-bit%8));
            }
            ++count;
        }
    }
    return count;
}

static PyObject *interval_layout(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *objects[3], *result = NULL;
    Py_buffer v[3] = {{0}};
    double lower_cut, upper_cut;
    if (!PyArg_ParseTuple(args, "OOOdd:interval_layout", &objects[0], &objects[1],
                          &objects[2], &lower_cut, &upper_cut)) return NULL;
    if (!isfinite(lower_cut) || !isfinite(upper_cut) || lower_cut < 0. || upper_cut <= lower_cut) {
        PyErr_SetString(PyExc_ValueError, "finite cutoffs with 0 <= lower < upper required");
        return NULL;
    }
    const char *names[] = {"target", "left", "right"};
    for (int i = 0; i < 3; ++i) {
        if (PyObject_GetBuffer(objects[i], &v[i], PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) goto fail;
        if (!validate_buffer(&v[i], 1, names[i])) goto fail;
        const double *x = v[i].buf;
        for (Py_ssize_t j = 0; j < v[i].shape[0]; ++j) {
            if (!isfinite(x[j])) {
                PyErr_Format(PyExc_ValueError, "%s must contain only finite values", names[i]);
                goto fail;
            }
        }
    }
    Py_ssize_t nt = v[0].shape[0], na = v[1].shape[0];
    if (v[2].shape[0] != na) {
        PyErr_SetString(PyExc_ValueError, "arterial interval shapes differ");
        goto fail;
    }
    const double *left = v[1].buf, *right = v[2].buf;
    for (Py_ssize_t j = 0; j < na; ++j) {
        if (right[j] <= left[j]) {
            PyErr_SetString(PyExc_ValueError, "arterial intervals must have positive width");
            goto fail;
        }
    }
    if (na && nt > PY_SSIZE_T_MAX/na) {
        PyErr_SetString(PyExc_OverflowError, "interval domain exceeds addressable extent");
        goto fail;
    }
    Py_ssize_t size = nt*na, mask_size = size/8 + (size%8 != 0);
    result = PyTuple_New(5);
    if (!result) goto fail;
    PyObject *mask = PyBytes_FromStringAndSize(NULL, mask_size);
    if (!mask) goto fail;
    PyTuple_SET_ITEM(result, 4, mask);
    unsigned char *packed = (unsigned char *)PyBytes_AS_STRING(mask);
    memset(packed, 0, mask_size);
    Py_ssize_t count = enumerate_intervals(nt, na, v[0].buf, left, right,
                                           lower_cut, upper_cut, packed, NULL);
    char *out[4];
    for (int i = 0; i < 4; ++i) {
        Py_ssize_t width = i < 2 ? sizeof(Py_ssize_t) : sizeof(double);
        if (count > PY_SSIZE_T_MAX/width) {
            PyErr_SetString(PyExc_OverflowError, "active interval output exceeds addressable extent");
            goto fail;
        }
        PyObject *buffer = PyBytes_FromStringAndSize(NULL, count*width);
        if (!buffer) goto fail;
        PyTuple_SET_ITEM(result, i, buffer);
        out[i] = PyBytes_AS_STRING(buffer);
    }
    enumerate_intervals(nt, na, v[0].buf, left, right, lower_cut, upper_cut, packed, out);
    for (int i = 0; i < 3; ++i) PyBuffer_Release(&v[i]);
    return result;
fail:
    for (int i = 0; i < 3; ++i) if (v[i].obj) PyBuffer_Release(&v[i]);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *contract_moments(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *objects[8], *result = NULL;
    Py_buffer v[8] = {{0}};
    if (!PyArg_UnpackTuple(args, "contract_moments", 8, 8, &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4], &objects[5],
                          &objects[6], &objects[7])) return NULL;
    const char *names[] = {"local", "segments", "coefficients", "factors", "p0", "p1", "q0", "q1"};
    int ndim[] = {1, 1, 2, 1, 2, 2, 2, 2};
    for (int i = 0; i < 8; ++i) {
        if (PyObject_GetBuffer(objects[i], &v[i], PyBUF_FORMAT | PyBUF_STRIDES) < 0) goto fail;
        if (!validate_typed_buffer(&v[i], ndim[i], names[i], i == 1, i >= 4)) goto fail;
    }
    Py_ssize_t n = v[0].shape[0], na = v[2].shape[1];
    if (v[1].shape[0] != n || v[2].shape[0] != 4 || v[3].shape[0] != 4) {
        PyErr_SetString(PyExc_ValueError, "incompatible moment array shapes");
        goto fail;
    }
    Py_ssize_t strides[4][2];
    const double *cdf[4];
    for (int i = 0; i < 4; ++i) {
        if (v[4+i].shape[0] != 4 || v[4+i].shape[1] != n) {
            PyErr_SetString(PyExc_ValueError, "CDF shape must be four by interval count");
            goto fail;
        }
        cdf[i] = v[4+i].buf;
        for (int j = 0; j < 2; ++j) strides[i][j] = v[4+i].strides[j]/sizeof(double);
    }
    const Py_ssize_t *segments = v[1].buf;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (segments[i] < 0 || segments[i] >= na) {
            PyErr_SetString(PyExc_ValueError, "arterial segment index out of range");
            goto fail;
        }
    }
    if (n > PY_SSIZE_T_MAX/(Py_ssize_t)sizeof(double)) {
        PyErr_SetString(PyExc_OverflowError, "moment output exceeds addressable extent");
        goto fail;
    }
    result = PyTuple_New(2);
    if (!result) goto fail;
    char *out[2];
    for (int i = 0; i < 2; ++i) {
        PyObject *buffer = PyBytes_FromStringAndSize(NULL, n*sizeof(double));
        if (!buffer) goto fail;
        PyTuple_SET_ITEM(result, i, buffer);
        out[i] = PyBytes_AS_STRING(buffer);
    }
    const double *t = v[0].buf, *c = v[2].buf, *factors = v[3].buf;
    const int choose[4][4] = {{1,0,0,0}, {1,1,0,0}, {1,2,1,0}, {1,3,3,1}};
    for (Py_ssize_t i = 0; i < n; ++i) {
        double power_t[4] = {1., t[i], t[i]*t[i], pow(t[i], 3.)};
        double total = 0., roundoff = 0.;
        for (int k = 0; k < 4; ++k) {
            double power = 0., absolute = 0.;
            for (int j = k; j < 4; ++j) {
                double piece = ((c[(3-j)*na+segments[i]]*choose[j][k])*power_t[j-k])
                               * (k%2 ? -1. : 1.);
                power += piece;
                absolute += fabs(piece);
            }
            double p0 = cdf[0][k*strides[0][0]+i*strides[0][1]];
            double mass, sum;
            if (p0 > .5) {
                double q0 = cdf[2][k*strides[2][0]+i*strides[2][1]];
                double q1 = cdf[3][k*strides[3][0]+i*strides[3][1]];
                mass = q0-q1;
                sum = q0+q1;
            } else {
                double p1 = cdf[1][k*strides[1][0]+i*strides[1][1]];
                mass = p1-p0;
                sum = p1+p0;
            }
            double term = (power*factors[k])*mass;
            total += term;
            roundoff += (absolute*factors[k])*sum+fabs(term);
        }
        double error = (16*DBL_EPSILON)*roundoff;
        memcpy(out[0]+i*sizeof(double), &total, sizeof(double));
        memcpy(out[1]+i*sizeof(double), &error, sizeof(double));
    }
    for (int i = 0; i < 8; ++i) PyBuffer_Release(&v[i]);
    return result;
fail:
    for (int i = 0; i < 8; ++i) if (v[i].obj) PyBuffer_Release(&v[i]);
    Py_XDECREF(result);
    return NULL;
}

static PyMethodDef methods[] = {
    {"convolve", convolve, METH_VARARGS,
     "Integrate validated arterial and cubic-residue buffers into a separate output."},
    {"interval_layout", interval_layout, METH_VARARGS,
     "Return owned active-interval index, lag and packed-mask buffers."},
    {"contract_moments", contract_moments, METH_VARARGS,
     "Return owned cubic-moment values and roundoff estimates."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_convolution", "Checked polynomial convolution.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__convolution(void) {
    return PyModule_Create(&module);
}

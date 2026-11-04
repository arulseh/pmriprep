/* Adapted from SciPy. See NOTICE_SCIPY.md and LICENSE_SCIPY.txt. */
#include <float.h>
#include <math.h>

static double norm4(const double *x) {
    double value = 0.;
    for (int i = 0; i < 4; ++i) value += x[i]*x[i];
    return sqrt(value);
}

static double maximum(double a, double b) { return b > a ? b : a; }

static void negative_product(const double *V, const double *x, double *out) {
    for (int i = 0; i < 4; ++i) {
        double value = 0.;
        for (int j = 0; j < 4; ++j) value += V[4*i+j]*x[j];
        out[i] = -value;
    }
}

static void phi_derivative(double alpha, const double *suf, const double *s,
                           double radius, double *phi, double *derivative) {
    double quotient[4], denom[4];
    for (int i = 0; i < 4; ++i) {
        denom[i] = s[i]*s[i]+alpha;
        quotient[i] = suf[i]/denom[i];
    }
    double length = norm4(quotient), sum = 0.;
    for (int i = 0; i < 4; ++i) sum += (suf[i]*suf[i])/pow(denom[i], 3.);
    *phi = length-radius;
    *derivative = -sum/length;
}

int trust4(int m, const double *uf, const double *s, const double *V,
           double radius, int has_alpha, double initial_alpha, double rtol,
           int max_iter, double *out) {
    if (m < 0 || !isfinite(radius) || radius <= 0. || !isfinite(rtol) || rtol <= 0.
        || max_iter < 1 || (has_alpha && !isfinite(initial_alpha))) return -1;
    for (int i = 0; i < 4; ++i)
        if (!isfinite(uf[i]) || !isfinite(s[i]) || s[i] < 0.) return -1;
    for (int i = 0; i < 16; ++i) if (!isfinite(V[i])) return -1;
    double suf[4], x[4];
    for (int i = 0; i < 4; ++i) suf[i] = s[i]*uf[i];
    const int full_rank = m >= 4 && s[3] > (DBL_EPSILON*m)*s[0];
    if (full_rank) {
        for (int i = 0; i < 4; ++i) x[i] = uf[i]/s[i];
        negative_product(V, x, out);
        if (norm4(out) <= radius) { out[4] = 0.; return 0; }
    }
    double upper = norm4(suf)/radius, lower = 0., phi, derivative;
    if (full_rank) {
        phi_derivative(0., suf, s, radius, &phi, &derivative);
        lower = -phi/derivative;
    }
    double alpha = !has_alpha || (!full_rank && initial_alpha == 0.)
        ? maximum(.001*upper, sqrt(lower*upper)) : initial_alpha;
    int iterations = 0;
    for (int i = 0; i < max_iter; ++i) {
        if (alpha < lower || alpha > upper) alpha = maximum(.001*upper, sqrt(lower*upper));
        phi_derivative(alpha, suf, s, radius, &phi, &derivative);
        if (phi < 0.) upper = alpha;
        const double ratio = phi/derivative;
        lower = maximum(lower, alpha-ratio);
        alpha -= (phi+radius)*ratio/radius;
        iterations = i+1;
        if (fabs(phi) < rtol*radius) break;
    }
    for (int i = 0; i < 4; ++i) x[i] = suf[i]/(s[i]*s[i]+alpha);
    negative_product(V, x, out);
    const double adjustment = radius/norm4(out);
    for (int i = 0; i < 4; ++i) out[i] *= adjustment;
    out[4] = alpha;
    for (int i = 0; i < 5; ++i) if (!isfinite(out[i])) return -2;
    return iterations;
}

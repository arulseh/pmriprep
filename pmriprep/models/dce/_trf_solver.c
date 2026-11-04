/* Adapted from SciPy. See NOTICE_SCIPY.md and LICENSE_SCIPY.txt. */
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "_trf_evaluator.c"
#include "_trf_trust.c"

typedef void (*gesdd_fn)(char *, int *, int *, double *, int *, double *,
                         double *, int *, double *, int *, double *, int *, int *, int *);

static double dotn(const double *a, const double *b, int n) {
    double value = 0.;
    for (int i = 0; i < n; ++i) value += a[i]*b[i];
    return value;
}

static int finite_array(const double *a, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(a[i])) return 0;
    return 1;
}

static void multiply(const double *J, const double *x, double *out, int m) {
    for (int i = 0; i < m; ++i) out[i] = dotn(J+4*i, x, 4);
}

static double quadratic(const double *J, const double *g, const double *x,
                        const double *diag, int m, double *scratch) {
    multiply(J, x, scratch, m);
    double q = dotn(scratch, scratch, m), d = 0.;
    for (int j = 0; j < 4; ++j) d += (x[j]*diag[j])*x[j];
    q += d;
    return .5*q+dotn(x, g, 4);
}

static void quadratic_line(const double *J, const double *g, const double *s,
                           const double *diag, const double *s0, int m,
                           double *a, double *b, double *c, double *u, double *v) {
    multiply(J, s, v, m);
    double d = 0.;
    for (int j = 0; j < 4; ++j) d += (s[j]*diag[j])*s[j];
    *a = .5*(dotn(v, v, m)+d);
    *b = dotn(g, s, 4);
    *c = 0.;
    if (s0) {
        multiply(J, s0, u, m);
        *b += dotn(u, v, m);
        *c = .5*dotn(u, u, m)+dotn(g, s0, 4);
        double bd = 0., cd = 0.;
        for (int j = 0; j < 4; ++j) {
            bd += (s0[j]*diag[j])*s[j];
            cd += (s0[j]*diag[j])*s0[j];
        }
        *b += bd;
        *c += .5*cd;
    }
}

static void minimize_line(double a, double b, double lower, double upper, double c,
                          double *point, double *value) {
    double points[3] = {lower, upper, 0.};
    int n = 2;
    if (a != 0.) {
        const double extremum = -.5*b/a;
        if (lower < extremum && extremum < upper) points[n++] = extremum;
    }
    *point = points[0];
    *value = points[0]*(a*points[0]+b)+c;
    for (int i = 1; i < n; ++i) {
        const double y = points[i]*(a*points[i]+b)+c;
        if (y < *value) { *point = points[i]; *value = y; }
    }
}

static double to_bound(const double *x, const double *s, const double *lb,
                        const double *ub, int *hits) {
    double steps[4], minimum = INFINITY;
    for (int j = 0; j < 4; ++j) {
        steps[j] = s[j] == 0. ? INFINITY : fmax((lb[j]-x[j])/s[j], (ub[j]-x[j])/s[j]);
        if (steps[j] < minimum) minimum = steps[j];
    }
    if (hits) for (int j = 0; j < 4; ++j) {
        hits[j] = steps[j] == minimum ? (s[j] > 0.)-(s[j] < 0.) : 0;
    }
    return minimum;
}

static int intersection(const double *x, const double *s, double radius, double *positive) {
    const double a = dotn(s, s, 4), b = dotn(x, s, 4), c = dotn(x, x, 4)-radius*radius;
    if (a == 0. || c > 0.) return -1;
    const double d = sqrt(b*b-a*c), q = -(b+copysign(d, b));
    const double t1 = q/a, t2 = c/q;
    *positive = t1 < t2 ? t2 : t1;
    return 0;
}

static int bounded(const double *x, const double *lb, const double *ub) {
    for (int j = 0; j < 4; ++j) if (x[j] < lb[j] || x[j] > ub[j]) return 0;
    return 1;
}

static int select_native(int m, const double *x, const double *J, const double *diag,
                          const double *g, double *p, double *ph, const double *d,
                          double radius, const double *lb, const double *ub, double theta,
                          double *step, double *steph, double *predicted, double *scratch) {
    double trial[4];
    for (int j = 0; j < 4; ++j) trial[j] = x[j]+p[j];
    if (bounded(trial, lb, ub)) {
        memcpy(step, p, 4*sizeof(double));
        memcpy(steph, ph, 4*sizeof(double));
        *predicted = -quadratic(J, g, ph, diag, m, scratch);
        return 0;
    }
    int hits[4];
    const double stride = to_bound(x, p, lb, ub, hits);
    double rh[4], r[4], onbound[4];
    for (int j = 0; j < 4; ++j) {
        rh[j] = hits[j] ? -ph[j] : ph[j];
        r[j] = d[j]*rh[j];
        p[j] *= stride;
        ph[j] *= stride;
        onbound[j] = x[j]+p[j];
    }
    double totr;
    if (intersection(ph, rh, radius, &totr)) return -2;
    const double tob = to_bound(onbound, r, lb, ub, NULL);
    const double rs = fmin(tob, totr);
    const double rlower = rs > 0. ? (1.-theta)*stride/rs : 0.;
    const double rupper = rs > 0. ? (rs == tob ? theta*tob : totr) : -1.;
    double rv = INFINITY, a, b, c, point;
    if (rlower <= rupper) {
        quadratic_line(J, g, rh, diag, ph, m, &a, &b, &c, scratch, scratch+m);
        minimize_line(a, b, rlower, rupper, c, &point, &rv);
        for (int j = 0; j < 4; ++j) {
            rh[j] *= point;
            rh[j] += ph[j];
            r[j] = rh[j]*d[j];
        }
    }
    for (int j = 0; j < 4; ++j) { p[j] *= theta; ph[j] *= theta; }
    const double pv = quadratic(J, g, ph, diag, m, scratch);
    double agh[4], ag[4];
    for (int j = 0; j < 4; ++j) { agh[j] = -g[j]; ag[j] = d[j]*agh[j]; }
    const double agtr = radius/norm4(agh), agb = to_bound(x, ag, lb, ub, NULL);
    const double agupper = agb < agtr ? theta*agb : agtr;
    quadratic_line(J, g, agh, diag, NULL, m, &a, &b, &c, scratch, scratch+m);
    double av;
    minimize_line(a, b, 0., agupper, 0., &point, &av);
    for (int j = 0; j < 4; ++j) { agh[j] *= point; ag[j] *= point; }
    const double *chosen, *chosenh;
    int branch;
    if (pv < rv && pv < av) { chosen = p; chosenh = ph; *predicted = -pv; branch = 1; }
    else if (rv < pv && rv < av) { chosen = r; chosenh = rh; *predicted = -rv; branch = 2; }
    else { chosen = ag; chosenh = agh; *predicted = -av; branch = 3; }
    memcpy(step, chosen, 4*sizeof(double));
    memcpy(steph, chosenh, 4*sizeof(double));
    return branch;
}

int trf_step(int m, const double *x, const double *J, const double *diag, const double *g,
             double *p, double *ph, const double *d, double radius, const double *lb,
             const double *ub, double theta, double *out) {
    if (m < 1 || m > 100000) return -10;
    double *scratch = malloc((size_t)m*2*sizeof(double));
    if (!scratch) return -11;
    int code = select_native(m, x, J, diag, g, p, ph, d, radius, lb, ub, theta,
                             out, out+4, out+8, scratch);
    free(scratch);
    return code;
}

static int factorize(gesdd_fn fn, int m, const double *row, double *u, double *s,
                     double *vt, double *a, double *work, int lwork) {
    if (lwork != -1 && !finite_array(row, m*4)) return -1;
    int n = 4, lda = m, ldu = m, ldvt = 4, info, iwork[32];
    char job = 'S';
    for (int i = 0; i < m; ++i) for (int j = 0; j < 4; ++j) a[i+m*j] = row[4*i+j];
    fn(&job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    return info;
}

int trf_svd(void *function, int m, const double *row, double *u, double *s, double *vt) {
    if (!function || m < 4 || m > 100004 || !finite_array(row, m*4)) return -10;
    gesdd_fn fn = (gesdd_fn)function;
    double *a = malloc((size_t)m*4*sizeof(double));
    if (!a) return -11;
    double query;
    int info = factorize(fn, m, row, u, s, vt, a, &query, -1);
    if (info || !isfinite(query) || query < 1. || query > INT_MAX) { free(a); return -12; }
    int lwork = (int)query;
    double *work = malloc((size_t)lwork*sizeof(double));
    if (!work) { free(a); return -11; }
    info = factorize(fn, m, row, u, s, vt, a, work, lwork);
    free(work);
    free(a);
    return info;
}

static void gradient(const double *J, const double *f, int m, double *g) {
    for (int j = 0; j < 4; ++j) {
        double value = 0.;
        for (int i = 0; i < m; ++i) value += J[4*i+j]*f[i];
        g[j] = value;
    }
}

static void jacscale(const double *J, int m, double *scale, double *inverse, int initial) {
    for (int j = 0; j < 4; ++j) {
        double value = 0.;
        for (int i = 0; i < m; ++i) value += J[4*i+j]*J[4*i+j];
        value = sqrt(value);
        inverse[j] = initial ? (value == 0. ? 1. : value) : fmax(value, inverse[j]);
        scale[j] = 1./inverse[j];
    }
}

static void boundscale(const double *x, const double *g, const double *lb,
                        const double *ub, double *v, double *dv) {
    for (int j = 0; j < 4; ++j) {
        if (g[j] < 0.) { v[j] = ub[j]-x[j]; dv[j] = -1.; }
        else if (g[j] > 0.) { v[j] = x[j]-lb[j]; dv[j] = 1.; }
        else { v[j] = 1.; dv[j] = 0.; }
    }
}

int trf_solve(Context *context, const double *curve, double curve_scale, const double *initial,
               void *svd_function, double ftol, double xtol, double gtol, int max_nfev,
               double *out, double *info_out, double *trace) {
    if (!context || !svd_function || context->observations < 1 || context->observations > 100000
        || max_nfev < 1 || !isfinite(curve_scale) || curve_scale <= 0.
        || !isfinite(ftol) || !isfinite(xtol) || !isfinite(gtol) || ftol <= 0. || xtol <= 0. || gtol <= 0.) return -10;
    const int m = (int)context->observations, ma = m+4;
    const double lb[4] = {0., 0., 0., 0.}, ub[4] = {5., 5., 1., 1.};
    if (!finite_array(initial, 4) || !bounded(initial, lb, ub) || !finite_array(curve, m)) return -10;
    double *memory = calloc((size_t)(26*m+128), sizeof(double));
    if (!memory) return -11;
    double *f = memory, *fnew = f+m, *pred = fnew+m, *jacres = pred+m,
           *J = jacres+m, *Jh = J+4*m, *aug = Jh+4*m, *a = aug+4*ma,
           *u = a+4*ma, *scratch = u+4*ma;
    double x[4], g[4], scale[4], inverse[4], v[4], dv[4];
    memcpy(x, initial, 4*sizeof(double));
    for (int j = 0; j < 4; ++j) {
        const double lo = x[j], hi = ub[j]-x[j];
        if (lo <= fmin(hi, 1e-10)) x[j] = 1e-10;
        if (hi <= fmin(lo, 1e-10*fmax(1., fabs(ub[j])))) x[j] = ub[j]-1e-10*fmax(1., fabs(ub[j]));
    }
    evaluate(context, x, curve, curve_scale, 1, pred, f, J);
    if (!finite_array(f, m) || !finite_array(J, 4*m)) { free(memory); return -13; }
    gradient(J, f, m, g);
    jacscale(J, m, scale, inverse, 1);
    boundscale(x, g, lb, ub, v, dv);
    double initial_hat[4];
    for (int j = 0; j < 4; ++j) {
        if (dv[j] != 0.) v[j] *= inverse[j];
        initial_hat[j] = x[j]*inverse[j]/sqrt(v[j]);
    }
    double radius = norm4(initial_hat), alpha = 0., cost = .5*dotn(f, f, m), gnorm = 0.;
    if (radius == 0.) radius = 1.;
    gesdd_fn svd = (gesdd_fn)svd_function;
    double singular[4], vt[16], query;
    int code = factorize(svd, ma, aug, u, singular, vt, a, &query, -1);
    if (code || !isfinite(query) || query < 1. || query > INT_MAX) { free(memory); return -12; }
    const int lwork = (int)query;
    double *work = malloc((size_t)lwork*sizeof(double));
    if (!work) { free(memory); return -11; }
    int status = 0, nfev = 1, njev = 1, iteration = 0, trace_count = 0;
    while (1) {
        boundscale(x, g, lb, ub, v, dv);
        gnorm = 0.;
        for (int j = 0; j < 4; ++j) gnorm = fmax(gnorm, fabs(g[j]*v[j]));
        if (gnorm < gtol) status = 1;
        if (trace) {
            double *row = trace+16*trace_count;
            row[0] = nfev; row[1] = cost; row[2] = radius; row[3] = alpha; row[4] = gnorm;
            for (int j = 0; j < 4; ++j) { row[5+j] = x[j]; row[9+j] = inverse[j]; }
            row[13] = status; row[14] = njev; row[15] = iteration;
        }
        ++trace_count;
        if (status || nfev == max_nfev) break;
        double d[4], diag[4], gh[4], uf[4] = {0., 0., 0., 0.}, V[16];
        for (int j = 0; j < 4; ++j) {
            if (dv[j] != 0.) v[j] *= inverse[j];
            d[j] = sqrt(v[j])*scale[j];
            diag[j] = g[j]*dv[j]*scale[j];
            gh[j] = d[j]*g[j];
        }
        for (int i = 0; i < m; ++i) for (int j = 0; j < 4; ++j) Jh[4*i+j] = aug[4*i+j] = J[4*i+j]*d[j];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) aug[4*(m+i)+j] = i == j ? sqrt(diag[j]) : 0.;
        code = factorize(svd, ma, aug, u, singular, vt, a, work, lwork);
        if (code) { code = -12; break; }
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < m; ++i) uf[j] += u[i+ma*j]*f[i];
            for (int i = 0; i < 4; ++i) V[4*i+j] = vt[j+4*i];
        }
        const double theta = fmax(.995, 1.-gnorm);
        double actual = -1., xnew[4], cost_new = cost;
        while (actual <= 0. && nfev < max_nfev) {
            double trust[5], ph[4], step[4], steph[4], pstep[4], predicted;
            int nit = trust4(m, uf, singular, V, radius, 1, alpha, .01, 10, trust);
            if (nit < 0) { code = -14; break; }
            alpha = trust[4];
            for (int j = 0; j < 4; ++j) { ph[j] = trust[j]; pstep[j] = d[j]*ph[j]; }
            code = select_native(m, x, Jh, diag, gh, pstep, ph, d, radius, lb, ub, theta,
                                  step, steph, &predicted, scratch);
            if (code < 0) { code = -15; break; }
            code = 0;
            for (int j = 0; j < 4; ++j) {
                xnew[j] = x[j]+step[j];
                if (xnew[j] <= lb[j]) xnew[j] = nextafter(lb[j], ub[j]);
                else if (xnew[j] >= ub[j]) xnew[j] = nextafter(ub[j], lb[j]);
                if (xnew[j] < lb[j] || xnew[j] > ub[j]) xnew[j] = .5*(lb[j]+ub[j]);
            }
            evaluate(context, xnew, curve, curve_scale, 0, pred, fnew, J);
            ++nfev;
            const double hnorm = norm4(steph);
            if (!finite_array(fnew, m)) { radius = .25*hnorm; continue; }
            cost_new = .5*dotn(fnew, fnew, m);
            actual = cost-cost_new;
            const double ratio = predicted > 0. ? actual/predicted : ((predicted == 0. && actual == 0.) ? 1. : 0.);
            double radius_new = radius;
            if (ratio < .25) radius_new = .25*hnorm;
            else if (ratio > .75 && hnorm > .95*radius) radius_new = 2.*radius;
            const int fstop = actual < ftol*cost && ratio > .25;
            const int xstop = norm4(step) < xtol*(xtol+norm4(x));
            status = fstop && xstop ? 4 : (fstop ? 2 : (xstop ? 3 : 0));
            if (status) break;
            alpha *= radius/radius_new;
            radius = radius_new;
        }
        if (code < 0) break;
        if (actual > 0.) {
            memcpy(x, xnew, sizeof(x));
            memcpy(f, fnew, (size_t)m*sizeof(double));
            cost = cost_new;
            evaluate(context, x, curve, curve_scale, 1, pred, jacres, J);
            ++njev;
            if (!finite_array(J, 4*m)) { code = -13; break; }
            gradient(J, f, m, g);
            jacscale(J, m, scale, inverse, 0);
        }
        ++iteration;
    }
    memcpy(out, x, sizeof(x));
    info_out[0] = status; info_out[1] = nfev; info_out[2] = njev; info_out[3] = iteration;
    info_out[4] = cost; info_out[5] = gnorm; info_out[6] = trace_count;
    free(work);
    free(memory);
    return code;
}

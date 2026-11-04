#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef double (*hyper_fn)(double, double, double, int);

typedef struct {
    size_t frames, observations, unique;
    double h, baseline, eps_flow, eps_ve, eps_vp, eps_eigen;
    const double *full, *part, *s, *q, *held, *tail, *unique_s;
    const int64_t *k, *inverse;
    double *moments, *states, *high_curve, *low_curve, *scratch;
    hyper_fn hyper;
    const double *interval_widths;
    const int64_t *interval_inverse;
} Context;

double bridge(Context *c, double a, double b, double z) {
    return c->hyper(a, b, z, 1);
}

static void convolution(Context *c, double rate, double *out) {
    const size_t u = c->unique + 1;
    for (size_t m = 0; m < 4; ++m) {
        c->moments[m*u] = c->hyper(1., (double)m+2., -(rate*c->h), 1)/((double)m+1.);
        for (size_t j = 0; j < c->unique; ++j) {
            c->moments[m*u+j+1] = c->hyper(1., (double)m+2., -(rate*c->unique_s[j]), 1)/((double)m+1.);
        }
    }
    const double decay = exp(-rate*c->h);
    double previous = 0.;
    c->states[0] = 0.;
    for (size_t j = 0; j < c->frames-1; ++j) {
        const size_t index = c->interval_widths ? 1+(size_t)c->interval_inverse[j] : 0;
        double increment = c->moments[index]*c->full[j];
        for (size_t m = 1; m < 4; ++m) {
            increment += c->moments[m*u+index]*c->full[m*(c->frames-1)+j];
        }
        const double interval_decay = c->interval_widths ? exp(-rate*c->interval_widths[j]) : decay;
        previous = increment+interval_decay*previous;
        c->states[j+1] = previous;
    }
    for (size_t j = 0; j < c->observations; ++j) {
        double value = exp(-rate*c->s[j])*c->states[c->k[j]];
        double partial = 0.;
        for (size_t m = 0; m < 4; ++m) {
            partial += c->moments[m*u+1+(size_t)c->inverse[j]]*c->part[m*c->observations+j];
        }
        value += partial;
        if (c->q[j] < 0.) value = 0.;
        if (c->baseline != 0.) {
            const double moment = c->hyper(1., 2., -(rate*c->held[j]), 1);
            value += ((c->baseline*exp(-rate*c->tail[j]))*c->held[j])*moment;
        }
        out[j] = value;
    }
}

static void predict(Context *c, const double *x, double *out) {
    const double fp = 100.*x[0];
    const double ps_physical = 100.*x[1];
    const double ve_physical = x[2];
    const double vp_physical = (1.-x[2])*x[3];
    const double flow = fmax(fp, c->eps_flow)/100.;
    const double ps = fmax(ps_physical, 0.)/100.;
    const double ve = fmax(ve_physical, c->eps_ve);
    const double vp = fmax(vp_physical, c->eps_vp);
    const double pe = ps/vp, ep = ps/ve, outflow = flow/vp;
    const double trace = outflow+pe+ep;
    double high = (trace+sqrt(fmax(trace*trace-(4.*outflow)*ep, 0.)))/2.;
    double low = high > 0. ? (outflow*ep)/high : 0.;
    high = fmax(high, c->eps_eigen);
    low = fmax(low, c->eps_eigen);
    const int degenerate = high-low < c->eps_eigen;
    if (degenerate) low = high;
    const double difference = fmax(high-low, c->eps_eigen);
    const double a1 = degenerate ? flow : flow*(high-ep-pe)/difference;
    const double a2 = degenerate ? 0. : flow*(ep+pe-low)/difference;
    convolution(c, high, c->high_curve);
    convolution(c, low, c->low_curve);
    for (size_t j = 0; j < c->observations; ++j) {
        out[j] = a1*c->high_curve[j]+a2*c->low_curve[j];
    }
    if (ve_physical <= 0.) {
        convolution(c, outflow, c->high_curve);
        for (size_t j = 0; j < c->observations; ++j) out[j] = flow*c->high_curve[j];
    }
}

void evaluate(Context *c, const double *x, const double *curve, double scale,
              int with_jacobian, double *prediction, double *residual, double *jacobian) {
    predict(c, x, prediction);
    for (size_t j = 0; j < c->observations; ++j) residual[j] = (prediction[j]-curve[j])/scale;
    if (!with_jacobian) return;
    const double upper[4] = {5., 5., 1., 1.};
    for (size_t p = 0; p < 4; ++p) {
        double shifted[4] = {x[0], x[1], x[2], x[3]};
        double delta = 1e-5*fmax(fabs(x[p]), .01);
        if (x[p]+delta > upper[p]) delta = -delta;
        shifted[p] += delta;
        predict(c, shifted, c->scratch);
        for (size_t j = 0; j < c->observations; ++j) {
            jacobian[j*4+p] = ((c->scratch[j]-prediction[j])/delta)/scale;
        }
    }
}

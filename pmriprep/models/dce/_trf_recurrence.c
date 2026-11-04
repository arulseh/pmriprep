#include <stddef.h>

void batch_recurrence(const double *increments, const double *decay,
                      double *states, size_t n_rates, size_t n_steps) {
    for (size_t i = 0; i < n_rates; ++i) {
        double previous = 0.0;
        states[i * (n_steps + 1)] = 0.0;
        for (size_t j = 0; j < n_steps; ++j) {
            previous = increments[i * n_steps + j] + decay[i] * previous;
            states[i * (n_steps + 1) + j + 1] = previous;
        }
    }
}

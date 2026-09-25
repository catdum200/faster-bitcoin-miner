/* Order statistics and bootstrap CIs for the interleaved benchmarks. */
#include "stats.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int fbm_cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Quantile of a sorted array, linear interpolation. */
double fbm_quantile(const double *sorted, int n, double q)
{
    double pos = q * (n - 1);
    int i = (int)pos;
    if (i >= n - 1)
        return sorted[n - 1];
    return sorted[i] + (pos - i) * (sorted[i + 1] - sorted[i]);
}

double fbm_median(const double *v, int n)
{
    double tmp[256];
    memcpy(tmp, v, sizeof(double) * n);
    qsort(tmp, n, sizeof(double), fbm_cmp_double);
    return fbm_quantile(tmp, n, 0.5);
}

/* Percentile bootstrap 95% CI of the median of v (resampling runs). */
void fbm_bootstrap_ci(const double *v, int n, double *lo, double *hi)
{
    enum { B = 2000 };
    static double meds[B];
    double tmp[256];
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            tmp[i] = v[s % (uint64_t)n];
        }
        meds[b] = fbm_median(tmp, n);
    }
    qsort(meds, B, sizeof(double), fbm_cmp_double);
    *lo = fbm_quantile(meds, B, 0.025);
    *hi = fbm_quantile(meds, B, 0.975);
}

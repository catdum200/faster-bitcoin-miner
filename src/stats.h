/* Order statistics and bootstrap CIs for the interleaved benchmarks. */
#ifndef FBM_STATS_H
#define FBM_STATS_H

/* qsort comparator for doubles. */
int fbm_cmp_double(const void *a, const void *b);
/* Quantile of a sorted array, linear interpolation. */
double fbm_quantile(const double *sorted, int n, double q);
/* Median of v (n <= 256), v unchanged. */
double fbm_median(const double *v, int n);
/* Percentile bootstrap 95% CI of the median of v (n <= 256). */
void fbm_bootstrap_ci(const double *v, int n, double *lo, double *hi);

#endif

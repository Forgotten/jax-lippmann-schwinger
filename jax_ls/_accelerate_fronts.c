// The front factorisation of `jax_ls.nd_solver.factorize_host`, one level of
// the dissection tree per call, with Apple Accelerate's LAPACK.  Called from
// `jax_ls.accelerate` through ctypes; macOS 13.3 or later (the new LAPACK
// interface).
//
// For each front i of a level, with row-major complex128 blocks
//   F_SS (s x s), F_SB (s x b), F_BS (b x s):
//   inverse = F_SS^{-1}, upper = F_SS^{-1} F_SB, lower = F_BS F_SS^{-1},
// with F_SS factored by partial pivoting (zgetrf) and the three blocks formed
// by triangular solves (zgetrs) -- exactly the SciPy path, so the pivots, and
// the growth and pivot-ratio diagnostics, are the same.  Fronts are processed
// in parallel with Grand Central Dispatch; LAPACK threads the large ones.

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef __LAPACK_double_complex zc;

// valid: k x s flags, nonzero for real (unpadded) separator points.
// growth, pivot_ratio: k outputs.
// Returns 0, or 1 + the index of the first front whose F_SS is exactly
// singular, or -1 when out of memory.
int acc_factor_fronts(int k, int s, int b, const zc *f_ss, const zc *f_sb,
                      const zc *f_bs, const unsigned char *valid, zc *inverse,
                      zc *upper, zc *lower, double *growth,
                      double *pivot_ratio) {
  __block int failed = 0;
  dispatch_apply((size_t)k, DISPATCH_APPLY_AUTO, ^(size_t i) {
    const zc *ss = f_ss + i * (size_t)s * s;
    const zc *sb = f_sb + i * (size_t)s * b;
    const unsigned char *ok = valid + i * (size_t)s;
    zc *inv = inverse + i * (size_t)s * s;
    zc *up = upper + i * (size_t)s * b;
    zc *lo = lower + i * (size_t)b * s;

    zc *lu = malloc((size_t)s * s * sizeof(zc));
    zc *work = malloc((size_t)s * (b > 0 ? b : 1) * sizeof(zc));
    __LAPACK_int *pivots = malloc((size_t)s * sizeof(__LAPACK_int));
    if (!lu || !work || !pivots) {
      free(lu), free(work), free(pivots);
      failed = -1;
      return;
    }

    // F_SS to column-major, and its largest entry on the valid points.
    double largest = 0.0;
    for (int r = 0; r < s; ++r)
      for (int c = 0; c < s; ++c) {
        zc value = ss[(size_t)r * s + c];
        lu[(size_t)c * s + r] = value;
        if (ok[r] && ok[c] && cabs(value) > largest) largest = cabs(value);
      }

    __LAPACK_int n = s, nb = b, info = 0;
    zgetrf_(&n, &n, lu, &n, pivots, &info);
    if (info > 0) {
      if (!failed) failed = (int)i + 1;
      free(lu), free(work), free(pivots);
      return;
    }

    // inverse: F_SS^{-T} column-major is F_SS^{-1} row-major.
    memset(inv, 0, (size_t)s * s * sizeof(zc));
    for (int r = 0; r < s; ++r) inv[(size_t)r * s + r] = 1.0;
    zgetrs_("T", &n, &n, lu, &n, pivots, inv, &n, &info);

    if (b > 0) {
      // upper = F_SS^{-1} F_SB, solved column-major, written back row-major.
      for (int r = 0; r < s; ++r)
        for (int c = 0; c < b; ++c)
          work[(size_t)c * s + r] = sb[(size_t)r * b + c];
      zgetrs_("N", &n, &nb, lu, &n, pivots, work, &n, &info);
      for (int r = 0; r < s; ++r)
        for (int c = 0; c < b; ++c)
          up[(size_t)r * b + c] = work[(size_t)c * s + r];

      // lower^T = F_SS^{-T} F_BS^T; F_BS row-major is F_BS^T column-major,
      // and the solution column-major is lower row-major: solve in place.
      memcpy(lo, f_bs + i * (size_t)b * s, (size_t)b * s * sizeof(zc));
      zgetrs_("T", &n, &nb, lu, &n, pivots, lo, &n, &info);
    }

    double top = 0.0, small = INFINITY, large = 0.0;
    for (int c = 0; c < s; ++c) {
      for (int r = 0; r <= c; ++r) {
        double value = cabs(lu[(size_t)c * s + r]);
        if (value > top) top = value;
      }
      if (ok[c]) {
        double d = cabs(lu[(size_t)c * s + c]);
        if (d < small) small = d;
        if (d > large) large = d;
      }
    }
    growth[i] = top / (largest > 1e-300 ? largest : 1e-300);
    pivot_ratio[i] = large > 0.0 ? small / large : 1.0;
    free(lu), free(work), free(pivots);
  });
  return failed;
}

/****************************************************************************
 *
 * MODULE:       i.sar.interferometry
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Per-pixel preparation of the minimum cost flow phase
 *               unwrapping: wrapped phase gradients, their statistical cost
 *               weights and the residues. Compiled both as C99 (host,
 *               OpenMP) and as OpenCL C (device).
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#ifndef SARUNWRAP_CORE_H
#define SARUNWRAP_CORE_H

#ifdef __OPENCL_VERSION__
#define GLOBAL __global
typedef char sarunwrap_s8;
/* Plain functions: drivers may decline to inline a C99 "inline" function,
 * which then has no symbol to link (Mesa Clover). */
#define INLINE
#else
#include <math.h>
#define GLOBAL
#define INLINE static inline
typedef signed char sarunwrap_s8;
#endif

#define SARUNWRAP_PI        3.14159265358979323846f
#define SARUNWRAP_TWO_PI    6.28318530717958647692f
/* Coherence bounds of the phase noise model. */
#define SARUNWRAP_GAMMA_MIN 0.02f
#define SARUNWRAP_GAMMA_MAX 0.995f

/* Wrapped difference b - a in [-pi, pi); 0 when either phase is null, so
 * that edges and residues touching null pixels are consistent. */
INLINE float sarunwrap_delta(float a, float b)
{
    float d;

    if (isnan(a) || isnan(b))
        return 0.0f;
    d = b - a;
    d -= SARUNWRAP_TWO_PI * floor((d + SARUNWRAP_PI) / SARUNWRAP_TWO_PI);
    return d;
}

/* Cost weight 1 / (2 sigma^2) of the true gradient around the wrapped one,
 * sigma^2 = (1 - g^2) / (2 L g^2) being the phase variance of L looks at
 * coherence g (Cramer-Rao bound). Edges touching null pixels are free. */
INLINE float sarunwrap_weight(float pa, float pb, float ga, float gb,
                              float looks)
{
    float g;

    if (isnan(pa) || isnan(pb))
        return 0.0f;
    ga = isnan(ga) ? 0.0f : ga;
    gb = isnan(gb) ? 0.0f : gb;
    g = 0.5f * (ga + gb);
    g = g < SARUNWRAP_GAMMA_MIN ? SARUNWRAP_GAMMA_MIN : g;
    g = g > SARUNWRAP_GAMMA_MAX ? SARUNWRAP_GAMMA_MAX : g;
    return looks * g * g / (1.0f - g * g);
}

/* Pixel (i, j): gradient to the right (dx, wx), gradient down (dy, wy) and
 * residue of the loop (i, j), (i, j+1), (i+1, j+1), (i+1, j) in units of
 * 2 pi: dx(i, j) + dy(i, j+1) - dx(i+1, j) - dy(i, j). */
INLINE void sarunwrap_prep(int i, int j, int rows, int cols,
                           GLOBAL const float *phase, GLOBAL const float *coh,
                           float looks, GLOBAL float *dx, GLOBAL float *wx,
                           GLOBAL float *dy, GLOBAL float *wy,
                           GLOBAL sarunwrap_s8 *q)
{
    long k = (long)i * cols + j;
    float p = phase[k], s;

    if (j < cols - 1) {
        long e = (long)i * (cols - 1) + j;

        dx[e] = sarunwrap_delta(p, phase[k + 1]);
        wx[e] = sarunwrap_weight(p, phase[k + 1], coh[k], coh[k + 1], looks);
    }
    if (i < rows - 1) {
        dy[k] = sarunwrap_delta(p, phase[k + cols]);
        wy[k] =
            sarunwrap_weight(p, phase[k + cols], coh[k], coh[k + cols], looks);
    }
    if (i < rows - 1 && j < cols - 1) {
        s = sarunwrap_delta(p, phase[k + 1]) +
            sarunwrap_delta(phase[k + 1], phase[k + cols + 1]) -
            sarunwrap_delta(phase[k + cols], phase[k + cols + 1]) -
            sarunwrap_delta(p, phase[k + cols]);
        q[(long)i * (cols - 1) + j] =
            (sarunwrap_s8)floor(s / SARUNWRAP_TWO_PI + 0.5f);
    }
}

#endif /* SARUNWRAP_CORE_H */

/****************************************************************************
 *
 * MODULE:       i.sar.interferometry
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Public interface of libsarunwrap, the minimum cost flow
 *               phase unwrapping library of i.sar.interferometry, loaded
 *               from Python through ctypes.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#ifndef SARUNWRAP_H
#define SARUNWRAP_H

#include <stdint.h>

/* Mirrored by the ctypes structure UnwrapJob of i.sar.interferometry.py:
 * keep both in the same order. */
struct sarunwrap_job {
    int rows, cols;
    const float *phase;     /* Wrapped phase (radians), NaN for null. */
    const float *coherence; /* Coherence, NaN for unknown. */
    float looks;            /* Number of looks of the phase noise model. */
    float *unwrapped;       /* Output unwrapped phase, NaN for null. */
    int32_t *component;     /* Output connected component, 0 for null. */
    int64_t stats[4];       /* Output: residues, components, augmenting
                               paths, largest edge correction (cycles). */
};

/* Select the device of the per-pixel preparation: "host" (C, OpenMP),
 * "auto" (first OpenCL GPU, then OpenCL CPU, then host), "gpu" or "cpu"
 * (OpenCL only); platform, when not NULL or empty, restricts OpenCL to the
 * platforms whose name contains it. The network flow solution always runs
 * on the host. Returns 0 on success with the device in msg, nonzero with
 * the error in msg. */
int sarunwrap_init(const char *device, const char *platform, int nthreads,
                   char *msg, int msglen);

/* Unwrap one image. Returns 0 on success, nonzero with the error in msg. */
int sarunwrap_run(struct sarunwrap_job *job, char *msg, int msglen);

void sarunwrap_finish(void);

#endif /* SARUNWRAP_H */

/* OpenCL entry point of the unwrapping preparation of i.sar.interferometry;
 * the per-pixel code is sarunwrap_core.h, prepended to this file when the
 * kernels are embedded. */

__kernel void prepare(int rows, int cols, __global const float *phase,
                      __global const float *coh, float looks,
                      __global float *dx, __global float *wx,
                      __global float *dy, __global float *wy,
                      __global sarunwrap_s8 *q)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j >= cols || i >= rows)
        return;
    sarunwrap_prep(i, j, rows, cols, phase, coh, looks, dx, wx, dy, wy, q);
}

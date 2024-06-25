/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__powerpc__)

#include <altivec.h>   /* Required for the Power GCC built-ins  */

#include "distances_powerpc.h"

#include <cmath>

#define FLOAT_VEC_SIZE 4
#define INT32_VEC_SIZE 4
#define INT8_VEC_SIZE  16

namespace faiss {

float
fvec_L2sqr_ref_ppc(const float* x, const float* y, size_t d) {
    size_t i;

    float res = 0;
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       for (i = 0; i < d; i++) {
           const float tmp = x[i] - y[i];
           res += tmp * tmp;
       }
       return res;
    */

    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;

    vector float vx, vy;
    vector float vtmp = {0, 0, 0, 0};
    vector float vres = {0, 0, 0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        /* Load up the data vectors */
        vx = vec_xl (i*sizeof(float), x);
        vy = vec_xl (i*sizeof(float), y);

        vtmp = vec_sub(vx, vy);
        vres = vec_madd(vtmp, vtmp, vres);
    }

    /* Handle any remaining data elements */
    for (i = base; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }

    return res + vres[0] + vres[1] + vres[2] + vres[3];
}

float
fvec_L1_ref_ppc(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       for (i = 0; i < d; i++) {
           const float tmp = x[i] - y[i];
           res += std::fabs(tmp);
       }
       return res;
    */

    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;

    vector float vx, vy;
    vector float vtmp = {0, 0, 0, 0};
    vector float vres = {0, 0, 0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);
        vy = vec_xl (i*sizeof(float), y);

        vtmp = vec_sub(vx, vy);
        vtmp = vec_abs(vtmp);
        vres = vec_add(vtmp, vres);
    }

    /* Handle any remaining data elements */
    for (i = base; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += std::fabs(tmp);
    }

    return res + vres[0] + vres[1] + vres[2] + vres[3];
}

float
fvec_Linf_ref_ppc(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       for (i = 0; i < d; i++) {
         res = std::fmax(res, std::fabs(x[i] - y[i]));
       }
       return res;
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;

    vector float vx, vy;
    vector float vtmp = {0, 0, 0, 0};
    vector float vres = {0, 0, 0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);
        vy = vec_xl (i*sizeof(float), y);

        vtmp = vec_sub(vx, vy);
        vtmp = vec_abs(vtmp);
	res = std::fmax(res, vtmp[0]);
	res = std::fmax(res, vtmp[1]);
	res = std::fmax(res, vtmp[2]);
	res = std::fmax(res, vtmp[3]);
    }

    /* Handle any remaining data elements */
    for (i = base; i < d; i++) {
        res = std::fmax(res, std::fabs(x[i] - y[i]));
    }

    return res;
}

float
fvec_inner_product_ref_ppc(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       for (i = 0; i < d; i++) {
           res += x[i] * y[i];
       }
       return res;
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;

    vector float vx, vy;
    vector float vres = {0, 0, 0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);
        vy = vec_xl (i*sizeof(float), y);

        vres = vec_madd(vx, vy, vres);
    }

    /* Handle any remaining data elements */
    for (i = base; i < d; i++) {
        res += x[i] * y[i];
    }
    return res + vres[0] + vres[1] + vres[2] + vres[3];
}

float
fvec_norm_L2sqr_ref_ppc(const float* x, size_t d) {
    size_t i;
    double res = 0;
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Note, the original code calculated res as a double precision value,
       than returned the result as a float.
       Original code:

       for (i = 0; i < d; i++) {
           res += x[i] * x[i];
       }
       return res;
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  Do the
       operation as double, then return result as a float.  If the input array
       size is not a power of FLOAT_VEC_SIZE, do the remaining elements in
       scalar mode.  */
    size_t base;

    vector float vx;
    vector double vxde, vxdo;
    vector double vtmpo = {0, 0}, vtmpe = {0, 0};
    vector double vreso = {0, 0}, vrese = {0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);

        /* Convert even/odd floats to double then square elements. */
        vxdo = vec_doubleo (vx);
        vtmpo = vec_mul (vxdo, vxdo);
        vreso = vec_add (vreso, vtmpo);
        vxde = vec_doublee (vx);
        vtmpe = vec_mul (vxde, vxde);
        vrese = vec_add (vrese, vtmpe);
    }

    /* Handle any remaining data elements */
    for (i = base; i < d; i++) {
        res += x[i] * x[i];
    }
    return res + vreso[0] + vreso[1] + vrese[0] + vrese[1];
}

void
fvec_L2sqr_ny_ref_ppc(float* dis, const float* x, const float* y, size_t d,
                      size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref_ppc(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_ref_ppc(float* ip, const float* x, const float* y,
                               size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_ref_ppc(x, y, d);
        y += d;
    }
}

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors. squared lengths of y should be provided as well
void
fvec_L2sqr_ny_transposed_ref_ppc(float* __restrict dis,
                                 const float* __restrict x,
                                 const float* __restrict y,
                                 const float* __restrict y_sqlen,
                                 size_t d, size_t d_offset, size_t ny) {
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       float x_sqlen = 0;
       for (size_t j = 0; j < d; j++) {
           x_sqlen += x[j] * x[j];
       }

       for (size_t i = 0; i < ny; i++) {
           float dp = 0;
           for (size_t j = 0; j < d; j++) {
               dp += x[j] * y[i + j * d_offset];
           }

           dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
       }
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;

    float x_sqlen = 0;
    vector float vx, vy, vy_sqlen, vdp = {0, 0, 0, 0};
    vector float vx_sqlen = {0, 0, 0, 0};
    vector float vres = {0, 0, 0, 0};
    vector float vzero = {0, 0, 0, 0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);
        vx_sqlen = vec_madd(vx, vx, vx_sqlen);
    }

    x_sqlen = vx_sqlen[0] + vx_sqlen[1] + vx_sqlen[2] + vx_sqlen[3];

    /* Handle any remaining x data elements, in scalar mode. */
    for (size_t j = base; j < d; j++) {
        x_sqlen += x[j + base] * x[j + base];
    }

    for (size_t i = 0; i < ny; i++ ) {
        float dp = 0;
        vdp = vzero;

        /* Unrolling gives better performance then trying to vectorize.  */
        base = (d / 16) * 16;
        for (size_t j = 0; j < base; j = j + 16) {
            dp += x[j] * y[i + j * d_offset];
            dp += x[j + 1] * y[i + (j + 1) * d_offset];
            dp += x[j + 2] * y[i + (j + 2) * d_offset];
            dp += x[j + 3] * y[i + (j + 3) * d_offset];
            dp += x[j + 4] * y[i + (j + 4) * d_offset];
            dp += x[j + 5] * y[i + (j + 5) * d_offset];
            dp += x[j + 6] * y[i + (j + 6) * d_offset];
            dp += x[j + 7] * y[i + (j + 7) * d_offset];
            dp += x[j + 8] * y[i + (j + 8) * d_offset];
            dp += x[j + 9] * y[i + (j + 9) * d_offset];
            dp += x[j + 10] * y[i + (j + 10) * d_offset];
            dp += x[j + 11] * y[i + (j + 11) * d_offset];
            dp += x[j + 12] * y[i + (j + 12) * d_offset];
            dp += x[j + 13] * y[i + (j + 13) * d_offset];
            dp += x[j + 14] * y[i + (j + 14) * d_offset];
            dp += x[j + 15] * y[i + (j + 15) * d_offset];
        }

      for (size_t j = base; j < d; j++) {
	  dp += x[j] * y[i + j * d_offset];
      }

      dis[i] = x_sqlen +  y_sqlen[i] - 2 * dp;
    }
}

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_ref_ppc(float* __restrict distances_tmp_buffer,
                              const float* __restrict x,
                              const float* __restrict y, size_t d, size_t ny) {

    fvec_L2sqr_ny_ref_ppc(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors and return the index of the nearest vector.
/// squared lengths of y should be provided as well
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_y_transposed_ref_ppc(float* __restrict distances_tmp_buffer,
                                           const float* __restrict x,
                                           const float* __restrict y,
                                           const float* __restrict y_sqlen,
                                           size_t d, size_t d_offset,
					   size_t ny) {
    fvec_L2sqr_ny_transposed_ref_ppc(distances_tmp_buffer, x, y, y_sqlen, d,
                                     d_offset, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }

    return nearest_idx;
}

void
fvec_madd_ref_ppc(size_t n, const float* a, float bf, const float* b,
                  float* c) {
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       for (size_t i = 0; i < n; i++) {
           c[i] = a[i] + bf * b[i];
       }
   */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base;
    vector float va, vb, vc, vbf = {bf, bf, bf, bf};

    base = (n / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        va = vec_xl (i*sizeof(float), a);
        vb = vec_xl (i*sizeof(float), b);

        vc = vec_madd (vb, vbf, va);
        vec_xst (vc, i*sizeof(float), c);
    }

    /* Handle any remaining data elements */
    for (size_t i = base; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
}

int
fvec_madd_and_argmin_ref_ppc(size_t n, const float* a, float bf,
                             const float* b, float* c) {
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       float vmin = 1e20;
       int imin = -1;

       for (size_t i = 0; i < n; i++) {
           c[i] = a[i] + bf * b[i];
           if (c[i] < vmin) {
               vmin = c[i];
               imin = i;
            }
       }
       return imin;
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    vector float va, vb, vc;
    vector float vbf = {bf, bf, bf, bf};
    float vmin = 1.0e20;
    int imin = -1;
    size_t base;

    base = (n / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        va = vec_xl (i*sizeof(float), a);
        vb = vec_xl (i*sizeof(float), b);

	vc = vec_madd(vbf, vb, va);

        /* Checke each vector element */
        for (int j = 0; j < FLOAT_VEC_SIZE; j++) {
            if (vc[j] < vmin) {
                vmin = c[i + j];
                imin = i + j;
             }
        }
    }

    /* Handle any remaining data elements */
    for (size_t i = base; i < n; i++) {
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

void
fvec_L2sqr_batch_4_ref_ppc(const float* x, const float* y0, const float* y1,
                           const float* y2, const float* y3, const size_t d,
                           float& dis0, float& dis1, float& dis2, float& dis3) {

    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
      Original code:

      float d0 = 0;
      float d1 = 0;
      float d2 = 0;
      float d3 = 0;
      for (size_t i = 0; i < d; ++i) {
          const float q0 = x[i] - y0[i];
          const float q1 = x[i] - y1[i];
          const float q2 = x[i] - y2[i];
          const float q3 = x[i] - y3[i];
          d0 += q0 * q0;
          d1 += q1 * q1;
          d2 += q2 * q2;
          d3 += q3 * q3;
      }

      dis0 = d0;
      dis1 = d1;
      dis2 = d2;
      dis3 = d3;
    */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */
    size_t base, remainder;

    vector float vx, vy0, vy1, vy2, vy3;
    vector float vd0 = {0, 0, 0, 0};
    vector float vd1 = {0, 0, 0, 0};
    vector float vd2 = {0, 0, 0, 0};
    vector float vd3 = {0, 0, 0, 0};
    vector float vq0, vq1, vq2, vq3;
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;
    remainder = d % FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        /* Load up the data vectors */
        vx = vec_xl (i*sizeof(float), x);
        vy0 = vec_xl (i*sizeof(float), y0);
        vy1 = vec_xl (i*sizeof(float), y1);
        vy2 = vec_xl (i*sizeof(float), y2);
        vy3 = vec_xl (i*sizeof(float), y3);

        /* Replace scalar subtract with vector subtract built-in.  */
        vq0 = vec_sub(vx, vy0);
	vq1 = vec_sub(vx, vy1);
	vq2 = vec_sub(vx, vy2);
	vq3 = vec_sub(vx, vy3);

	/* Replace scalar multiply add with vector multiply add built-in.  */
	vd0 = vec_madd (vq0, vq0, vd0);
	vd1 = vec_madd (vq1, vq1, vd1);
	vd2 = vec_madd (vq2, vq2, vd2);
	vd3 = vec_madd (vq3, vq3, vd3);
    }

    /* Handle the remainder of the elments in scalar mode.  */
    for (size_t i = base; i < d ; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];

        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    /* Replace result assignment of the scalar result with sum of the
       corresponding vector elements to get the equivalent result.  */
    dis0 = vd0[0] + vd0[1] + vd0[2] + vd0[3] + d0;
    dis1 = vd1[0] + vd1[1] + vd1[2] + vd1[3] + d1;
    dis2 = vd2[0] + vd2[1] + vd2[2] + vd2[3] + d2;
    dis3 = vd3[0] + vd3[1] + vd3[2] + vd3[3] + d3;
}

void
fvec_inner_product_batch_4_ref_ppc(const float* __restrict x,
                                   const float* __restrict y0,
                                   const float* __restrict y1,
                                   const float* __restrict y2,
                                   const float* __restrict y3,
                                   const size_t d, float& dis0, float& dis1,
                                   float& dis2, float& dis3) {
    /* PowerPC, vectorize the function using PowerPC GCC built-in calls.
       Original code:

       float d0 = 0;
       float d1 = 0;
       float d2 = 0;
       float d3 = 0;
       for (size_t i = 0; i < d; ++i) {
           d0 += x[i] * y0[i];
           d1 += x[i] * y1[i];
           d2 += x[i] * y2[i];
           d3 += x[i] * y3[i];
       }

       dis0 = d0;
       dis1 = d1;
       dis2 = d2;
       dis3 = d3;
   */
    /* Vector implmentaion uses vector size of FLOAT_VEC_SIZE.  If the input
       array size is not a power of FLOAT_VEC_SIZE, do the remaining elements
       in scalar mode.  */

    size_t base, remainder;
    vector float vx, vy0, vy1, vy2, vy3;
    vector float vd0 = {0.0, 0.0, 0.0, 0.0};
    vector float vd1 = {0.0, 0.0, 0.0, 0.0};
    vector float vd2 = {0.0, 0.0, 0.0, 0.0};
    vector float vd3 = {0.0, 0.0, 0.0, 0.0};

    base = (d / FLOAT_VEC_SIZE) * FLOAT_VEC_SIZE;
    remainder = d % FLOAT_VEC_SIZE;

    for (size_t i = 0; i < base; i = i + FLOAT_VEC_SIZE) {
        vx = vec_xl (i*sizeof(float), x);
        vy0 = vec_xl (i*sizeof(float), y0);
        vy1 = vec_xl (i*sizeof(float), y1);
        vy2 = vec_xl (i*sizeof(float), y2);
        vy3 = vec_xl (i*sizeof(float), y3);

        vd0 = vec_madd (vx, vy0, vd0);
        vd1 = vec_madd (vx, vy1, vd1);
        vd2 = vec_madd (vx, vy2, vd2);
        vd3 = vec_madd (vx, vy3, vd3);
    }

    dis0 = vd0[0] + vd0[1] + vd0[2] + vd0[3];
    dis1 = vd1[0] + vd1[1] + vd1[2] + vd1[3];
    dis2 = vd2[0] + vd2[1] + vd2[2] + vd2[3];
    dis3 = vd3[0] + vd3[1] + vd3[2] + vd3[3];

    /* Handle any remaining data elements */
    if (remainder != 0) {
        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = 0;

        for (size_t i = base; i < d; i++) {
            d0 += x[i] * y0[i];
            d1 += x[i] * y1[i];
            d2 += x[i] * y2[i];
            d3 += x[i] * y3[i];
        }

        dis0 += d0;
        dis1 += d1;
        dis2 += d2;
        dis3 += d3;
    }
}

int32_t
ivec_inner_product_ref_ppc(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;

    /* Attempts to mannually vectorize and manually unroll the loop
       do not seem to improve the performance. */
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

int32_t
ivec_L2sqr_ref_ppc(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;

    /* Attempts to mannually vectorize and manually unroll the loop
       do not seem to improve the performance. */
    for (i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
	res += tmp * tmp;
    }
    return res;
}

}  // namespace faiss

#endif

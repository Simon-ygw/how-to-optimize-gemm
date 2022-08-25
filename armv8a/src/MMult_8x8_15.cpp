#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define GEMM_M 256
#define GEMM_N 256
#define GEMM_K 256

void Kernel_8x8(int m, int n, int k, float *sa, float *sb, float *sc, int ldc)
{
    
    float32x4_t a_vreg0, a_vreg1, b_vreg0, b_vreg1;

    float *bptr = sb;
    float *aptr = sa;
    float *cptr = sc;

    for (int i = 0; i < m; i += 8) {
        cptr = sc + ldc * i;
        for (int j = 0; j < n; j += 8) {

            __builtin_prefetch(aptr, 0, 3);
            __builtin_prefetch(bptr, 0, 3);

            float32x4_t c0{0}, c1{0}, c2{0}, c3{0}, c4{0}, c5{0}, c6{0}, c7{0},
                        c8{0}, c9{0}, c10{0}, c11{0}, c12{0}, c13{0}, c14{0}, c15{0};

            for (int p = 0; p < k; p += 8) {
                a_vreg0 = vld1q_f32(aptr);
                a_vreg1 = vld1q_f32(aptr + 4);
                b_vreg0 = vld1q_f32(bptr);
                b_vreg1 = vld1q_f32(bptr + 4);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 8);
                a_vreg1 = vld1q_f32(aptr + 12);
                b_vreg0 = vld1q_f32(bptr + 8);
                b_vreg1 = vld1q_f32(bptr + 12);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 16);
                a_vreg1 = vld1q_f32(aptr + 20);
                b_vreg0 = vld1q_f32(bptr + 16);
                b_vreg1 = vld1q_f32(bptr + 20);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 24);
                a_vreg1 = vld1q_f32(aptr + 28);
                b_vreg0 = vld1q_f32(bptr + 24);
                b_vreg1 = vld1q_f32(bptr + 28);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 32);
                a_vreg1 = vld1q_f32(aptr + 36);
                b_vreg0 = vld1q_f32(bptr + 32);
                b_vreg1 = vld1q_f32(bptr + 36);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 40);
                a_vreg1 = vld1q_f32(aptr + 44);
                b_vreg0 = vld1q_f32(bptr + 40);
                b_vreg1 = vld1q_f32(bptr + 44);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 48);
                a_vreg1 = vld1q_f32(aptr + 52);
                b_vreg0 = vld1q_f32(bptr + 48);
                b_vreg1 = vld1q_f32(bptr + 52);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                a_vreg0 = vld1q_f32(aptr + 56);
                a_vreg1 = vld1q_f32(aptr + 60);
                b_vreg0 = vld1q_f32(bptr + 56);
                b_vreg1 = vld1q_f32(bptr + 60);

                c0 = vmlaq_laneq_f32(c0, b_vreg0, a_vreg0, 0);
                c1 = vmlaq_laneq_f32(c1, b_vreg1, a_vreg0, 0);
                c2 = vmlaq_laneq_f32(c2, b_vreg0, a_vreg0, 1);
                c3 = vmlaq_laneq_f32(c3, b_vreg1, a_vreg0, 1);
                c4 = vmlaq_laneq_f32(c4, b_vreg0, a_vreg0, 2);
                c5 = vmlaq_laneq_f32(c5, b_vreg1, a_vreg0, 2);
                c6 = vmlaq_laneq_f32(c6, b_vreg0, a_vreg0, 3);
                c7 = vmlaq_laneq_f32(c7, b_vreg1, a_vreg0, 3);

                c8 = vmlaq_laneq_f32(c8, b_vreg0, a_vreg1, 0);
                c9 = vmlaq_laneq_f32(c9, b_vreg1, a_vreg1, 0);
                c10 = vmlaq_laneq_f32(c10, b_vreg0, a_vreg1, 1);
                c11 = vmlaq_laneq_f32(c11, b_vreg1, a_vreg1, 1);
                c12 = vmlaq_laneq_f32(c12, b_vreg0, a_vreg1, 2);
                c13 = vmlaq_laneq_f32(c13, b_vreg1, a_vreg1, 2);
                c14 = vmlaq_laneq_f32(c14, b_vreg0, a_vreg1, 3);
                c15 = vmlaq_laneq_f32(c15, b_vreg1, a_vreg1, 3);

                aptr += 64;
                bptr += 64;
            }

            c0 = vaddq_f32(c0, vld1q_f32(cptr));
            c1 = vaddq_f32(c1, vld1q_f32(cptr + 4));
            c2 = vaddq_f32(c2, vld1q_f32(cptr + ldc));
            c3 = vaddq_f32(c3, vld1q_f32(cptr + 4 + ldc));
            c4 = vaddq_f32(c4, vld1q_f32(cptr + 2 * ldc));
            c5 = vaddq_f32(c5, vld1q_f32(cptr + 4 + 2 * ldc));
            c6 = vaddq_f32(c6, vld1q_f32(cptr + 3 * ldc));
            c7 = vaddq_f32(c7, vld1q_f32(cptr + 4 + 3 * ldc));
            c8 = vaddq_f32(c8, vld1q_f32(cptr + 4 * ldc));
            c9 = vaddq_f32(c9, vld1q_f32(cptr + 4 + 4 * ldc));
            c10 = vaddq_f32(c10, vld1q_f32(cptr + 5 * ldc));
            c11 = vaddq_f32(c11, vld1q_f32(cptr + 4 + 5 * ldc));
            c12 = vaddq_f32(c12, vld1q_f32(cptr + 6 * ldc));
            c13 = vaddq_f32(c13, vld1q_f32(cptr + 4 + 6 * ldc));
            c14 = vaddq_f32(c14, vld1q_f32(cptr + 7 * ldc));
            c15 = vaddq_f32(c15, vld1q_f32(cptr + 4 + 7 * ldc));


            vst1q_f32(cptr, c0);
            vst1q_f32(cptr + 4, c1);
            vst1q_f32(cptr + ldc, c2);
            vst1q_f32(cptr + ldc + 4, c3);
            vst1q_f32(cptr + 2 * ldc, c4);
            vst1q_f32(cptr + 2 * ldc + 4, c5);
            vst1q_f32(cptr + 3 * ldc, c6);
            vst1q_f32(cptr + 3 * ldc + 4, c7);
            vst1q_f32(cptr + 4 * ldc, c8);
            vst1q_f32(cptr + 4 * ldc + 4, c9);
            vst1q_f32(cptr + 5 * ldc, c10);
            vst1q_f32(cptr + 5 * ldc + 4, c11);
            vst1q_f32(cptr + 6 * ldc, c12);
            vst1q_f32(cptr + 6 * ldc + 4, c13);
            vst1q_f32(cptr + 7 * ldc, c14);
            vst1q_f32(cptr + 7 * ldc + 4, c15);

            cptr += 8;
            aptr -= 8 * k;
        }

        bptr = sb;
        aptr += 8 * k;
    }
}

void PackMatrixA_8(int m, int k, float *a, int lda, float *a_to)
{ 
    float *a0_pntr, *a1_pntr, *a2_pntr, *a3_pntr, *a4_pntr, *a5_pntr, *a6_pntr, *a7_pntr;

    int ik = (k >> 3);

    for (int i = 0; i < m; i += 8) {
        a0_pntr = a + i * lda;
        a1_pntr = a0_pntr + lda;
        a2_pntr = a1_pntr + lda;
        a3_pntr = a2_pntr + lda;
        a4_pntr = a3_pntr + lda;
        a5_pntr = a4_pntr + lda;
        a6_pntr = a5_pntr + lda;
        a7_pntr = a6_pntr + lda;
        for (int j = 0; j < ik; j++) {

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
            *a_to++ = *a4_pntr++;
            *a_to++ = *a5_pntr++;
            *a_to++ = *a6_pntr++;
            *a_to++ = *a7_pntr++;
        }
    }
}

void PackMatrixB_8(int k, int n, float *b, int ldb, float *b_to) 
{
    float *btmp0, *btmp1, *btmp2, *btmp3, *btmp4, *btmp5, *btmp6, *btmp7;
    int in = (n >> 3);
    int ik = (k >> 3);

    for (int i = 0; i < in; i++) {
        btmp0 = b + (i << 3);
        for (int j = 0; j < ik; j++) {
            btmp1 = btmp0 + ldb;
            btmp2 = btmp0 + 2 * ldb;
            btmp3 = btmp0 + 3 * ldb;
            btmp4 = btmp0 + 4 * ldb;
            btmp5 = btmp0 + 5 * ldb;
            btmp6 = btmp0 + 6 * ldb;
            btmp7 = btmp0 + 7 * ldb;

            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;

            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;

            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;

            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;

            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;
            *b_to++ = *btmp4++;

            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;
            *b_to++ = *btmp5++;

            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;
            *b_to++ = *btmp6++;

            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;
            *b_to++ = *btmp7++;

            btmp0 += (8 * ldb - 8);
        }
    }
}



void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int pb, ib;

    float *packedA = (float*)aligned_alloc(64, m * k * sizeof(float));
    float *packedB = (float*)aligned_alloc(64, k * n * sizeof(float));

    int minM{GEMM_M}, minN{GEMM_N}, minK{GEMM_K};

    for (int i = 0; i < m; i += minM) {
        minM = std::min(GEMM_M, m - i);

        for (int p = 0; p < k; p += minK) {
            minK = std::min(GEMM_K, k - p);
        

            minN = std::min(n, GEMM_N);
            PackMatrixB_8(minK, minN, b + p * ldb, ldb, packedB);

            int minMM = minM;

            for (int mms = i; mms < i + minM; mms += minMM) {
                PackMatrixA_8(minMM, minK, a + mms * lda + p, lda, packedA + minK * (mms - i));

                Kernel_8x8(minMM, minN, minK, packedA + minK * (mms - i), packedB, c + mms * ldc, ldc);
            }

            for (int ns = minN; ns < n; ns += minN) {
                minN = std::min(n - ns, GEMM_N);
                PackMatrixB_8(minK, minN, b + ns + ldb * p, ldb, packedB);

                Kernel_8x8(minM, minN, minK, packedA, packedB, c + i * ldc + ns, ldc);
            }
        }
    }

    free(packedA);
    free(packedB);

}
#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define GEMM_M 1024
#define GEMM_N 256
#define GEMM_K 256

#define GEMM_UNROLL  8

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

            *a_to = *a0_pntr;
            *(a_to+1) = *a1_pntr;
            *(a_to+2) = *a2_pntr;
            *(a_to+3) = *a3_pntr;
            *(a_to+4) = *a4_pntr;
            *(a_to+5) = *a5_pntr;
            *(a_to+6) = *a6_pntr;
            *(a_to+7) = *a7_pntr;

            *(a_to+8) = *(a0_pntr + 1);
            *(a_to+9) = *(a1_pntr + 1);
            *(a_to+10) = *(a2_pntr + 1);
            *(a_to+11) = *(a3_pntr + 1);
            *(a_to+12) = *(a4_pntr + 1);
            *(a_to+13) = *(a5_pntr + 1);
            *(a_to+14) = *(a6_pntr + 1);
            *(a_to+15) = *(a7_pntr + 1);

            *(a_to+16) = *(a0_pntr + 2);
            *(a_to+17) = *(a1_pntr + 2);
            *(a_to+18) = *(a2_pntr + 2);
            *(a_to+19) = *(a3_pntr + 2);
            *(a_to+20) = *(a4_pntr + 2);
            *(a_to+21) = *(a5_pntr + 2);
            *(a_to+22) = *(a6_pntr + 2);
            *(a_to+23) = *(a7_pntr + 2);

            *(a_to+24) = *(a0_pntr + 3);
            *(a_to+25) = *(a1_pntr + 3);
            *(a_to+26) = *(a2_pntr + 3);
            *(a_to+27) = *(a3_pntr + 3);
            *(a_to+28) = *(a4_pntr + 3);
            *(a_to+29) = *(a5_pntr + 3);
            *(a_to+30) = *(a6_pntr + 3);
            *(a_to+31) = *(a7_pntr + 3);

            *(a_to+32) = *(a0_pntr + 4);
            *(a_to+33) = *(a1_pntr + 4);
            *(a_to+34) = *(a2_pntr + 4);
            *(a_to+35) = *(a3_pntr + 4);
            *(a_to+36) = *(a4_pntr + 4);
            *(a_to+37) = *(a5_pntr + 4);
            *(a_to+38) = *(a6_pntr + 4);
            *(a_to+39) = *(a7_pntr + 4);

            *(a_to+40) = *(a0_pntr + 5);
            *(a_to+41) = *(a1_pntr + 5);
            *(a_to+42) = *(a2_pntr + 5);
            *(a_to+43) = *(a3_pntr + 5);
            *(a_to+44) = *(a4_pntr + 5);
            *(a_to+45) = *(a5_pntr + 5);
            *(a_to+46) = *(a6_pntr + 5);
            *(a_to+47) = *(a7_pntr + 5);

            *(a_to+48) = *(a0_pntr + 6);
            *(a_to+49) = *(a1_pntr + 6);
            *(a_to+50) = *(a2_pntr + 6);
            *(a_to+51) = *(a3_pntr + 6);
            *(a_to+52) = *(a4_pntr + 6);
            *(a_to+53) = *(a5_pntr + 6);
            *(a_to+54) = *(a6_pntr + 6);
            *(a_to+55) = *(a7_pntr + 6);

            *(a_to+56) = *(a0_pntr + 7);
            *(a_to+57) = *(a1_pntr + 7);
            *(a_to+58) = *(a2_pntr + 7);
            *(a_to+59) = *(a3_pntr + 7);
            *(a_to+60) = *(a4_pntr + 7);
            *(a_to+61) = *(a5_pntr + 7);
            *(a_to+62) = *(a6_pntr + 7);
            *(a_to+63) = *(a7_pntr + 7);

            a_to += 64;
            a0_pntr += 8;
            a1_pntr += 8;
            a2_pntr += 8;
            a3_pntr += 8;
            a4_pntr += 8;
            a5_pntr += 8;
            a6_pntr += 8;
            a7_pntr += 8;
        }
    }
}

void PackMatrixB_8(int k, int n, float *b, int ldb, float *b_to) 
{
    float *btmp0, *btmp1, *btmp2, *btmp3, *btmp4, *btmp5, *btmp6, *btmp7;
    int in = (n >> 3);
    int ik = (k >> 3);
    float32x4x2_t v0,v1,v2,v3,v4,v5,v6,v7;

    //float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

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
#if 0
            t0 = *(btmp0+0);
            t1 = *(btmp0+1);
            t2 = *(btmp0+2);
            t3 = *(btmp0+3);
            t4 = *(btmp0+4);
            t5 = *(btmp0+5);
            t6 = *(btmp0+6);
            t7 = *(btmp0+7);

            t8 = *(btmp1+0);
            t9 = *(btmp1+1);
            t10 = *(btmp1+2);
            t11 = *(btmp1+3);
            t12 = *(btmp1+4);
            t13 = *(btmp1+5);
            t14 = *(btmp1+6);
            t15 = *(btmp1+7);

            *(b_to + 0) = t0;
            *(b_to + 1) = t1;
            *(b_to + 2) = t2;
            *(b_to + 3) = t3;
            *(b_to + 4) = t4;
            *(b_to + 5) = t5;
            *(b_to + 6) = t6;
            *(b_to + 7) = t7;

            *(b_to + 8) = t8;
            *(b_to + 9) = t9;
            *(b_to + 10) = t10;
            *(b_to + 11) = t11;
            *(b_to + 12) = t12;
            *(b_to + 13) = t13;
            *(b_to + 14) = t14;
            *(b_to + 15) = t15;

            b_to += 16;

            t0 = *(btmp2+0);
            t1 = *(btmp2+1);
            t2 = *(btmp2+2);
            t3 = *(btmp2+3);
            t4 = *(btmp2+4);
            t5 = *(btmp2+5);
            t6 = *(btmp2+6);
            t7 = *(btmp2+7);

            t8 = *(btmp3+0);
            t9 = *(btmp3+1);
            t10 = *(btmp3+2);
            t11 = *(btmp3+3);
            t12 = *(btmp3+4);
            t13 = *(btmp3+5);
            t14 = *(btmp3+6);
            t15 = *(btmp3+7);

            *(b_to + 0) = t0;
            *(b_to + 1) = t1;
            *(b_to + 2) = t2;
            *(b_to + 3) = t3;
            *(b_to + 4) = t4;
            *(b_to + 5) = t5;
            *(b_to + 6) = t6;
            *(b_to + 7) = t7;

            *(b_to + 8) = t8;
            *(b_to + 9) = t9;
            *(b_to + 10) = t10;
            *(b_to + 11) = t11;
            *(b_to + 12) = t12;
            *(b_to + 13) = t13;
            *(b_to + 14) = t14;
            *(b_to + 15) = t15;

            b_to += 16;

            t0 = *(btmp4+0);
            t1 = *(btmp4+1);
            t2 = *(btmp4+2);
            t3 = *(btmp4+3);
            t4 = *(btmp4+4);
            t5 = *(btmp4+5);
            t6 = *(btmp4+6);
            t7 = *(btmp4+7);

            t8 = *(btmp5+0);
            t9 = *(btmp5+1);
            t10 = *(btmp5+2);
            t11 = *(btmp5+3);
            t12 = *(btmp5+4);
            t13 = *(btmp5+5);
            t14 = *(btmp5+6);
            t15 = *(btmp5+7);

            *(b_to + 0) = t0;
            *(b_to + 1) = t1;
            *(b_to + 2) = t2;
            *(b_to + 3) = t3;
            *(b_to + 4) = t4;
            *(b_to + 5) = t5;
            *(b_to + 6) = t6;
            *(b_to + 7) = t7;

            *(b_to + 8) = t8;
            *(b_to + 9) = t9;
            *(b_to + 10) = t10;
            *(b_to + 11) = t11;
            *(b_to + 12) = t12;
            *(b_to + 13) = t13;
            *(b_to + 14) = t14;
            *(b_to + 15) = t15;

            b_to += 16;

            t0 = *(btmp6+0);
            t1 = *(btmp6+1);
            t2 = *(btmp6+2);
            t3 = *(btmp6+3);
            t4 = *(btmp6+4);
            t5 = *(btmp6+5);
            t6 = *(btmp6+6);
            t7 = *(btmp6+7);

            t8 = *(btmp7+0);
            t9 = *(btmp7+1);
            t10 = *(btmp7+2);
            t11 = *(btmp7+3);
            t12 = *(btmp7+4);
            t13 = *(btmp7+5);
            t14 = *(btmp7+6);
            t15 = *(btmp7+7);

            *(b_to + 0) = t0;
            *(b_to + 1) = t1;
            *(b_to + 2) = t2;
            *(b_to + 3) = t3;
            *(b_to + 4) = t4;
            *(b_to + 5) = t5;
            *(b_to + 6) = t6;
            *(b_to + 7) = t7;

            *(b_to + 8) = t8;
            *(b_to + 9) = t9;
            *(b_to + 10) = t10;
            *(b_to + 11) = t11;
            *(b_to + 12) = t12;
            *(b_to + 13) = t13;
            *(b_to + 14) = t14;
            *(b_to + 15) = t15;

            b_to += 16;
#else
            v0 = vld2q_f32(btmp0);
            v1 = vld2q_f32(btmp1);
            v2 = vld2q_f32(btmp2);
            v3 = vld2q_f32(btmp3);
            v4 = vld2q_f32(btmp4);
            v5 = vld2q_f32(btmp5);
            v6 = vld2q_f32(btmp6);
            v7 = vld2q_f32(btmp7);

            vst2q_f32(b_to, v0);
            vst2q_f32(b_to + 8, v1);
            vst2q_f32(b_to + 16, v2);
            vst2q_f32(b_to + 24, v3);
            vst2q_f32(b_to + 32, v4);
            vst2q_f32(b_to + 40, v5);
            vst2q_f32(b_to + 48, v6);
            vst2q_f32(b_to + 56, v7);
            b_to += 64;
#endif
            

            btmp0 += 8 * ldb;
        }
    }
}


void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float *packedA = (float*)aligned_alloc(64, m * k * sizeof(float));
    float *packedB = (float*)aligned_alloc(64, k * n * sizeof(float));

    int minM{GEMM_M}, minN{GEMM_N}, minK{GEMM_K};

    for (int i = 0; i < m; i += minM) {
        minM = std::min(GEMM_M, m - i);

        for (int p = 0; p < k; p += minK) {
            minK = k - p;

            if (minK > 2 * GEMM_K) {
                minK = GEMM_K;
            } else if (minK > GEMM_K) {
                minK = (minK / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            } 
        

            minN = n;
            if (minN > 2 * GEMM_N) {
                minN = GEMM_N;
            } else if (minN > GEMM_N) {
                minN = (minN / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }

            PackMatrixB_8(minK, minN, b + p * ldb, ldb, packedB);

            int minMM;

            for (int mms = i; mms < i + minM; mms += minMM) {
                minMM = i + minM - mms;
                if (minMM > 3 * GEMM_UNROLL) {
                    minMM = 3 * GEMM_UNROLL;
                } else if (minMM > 2 * GEMM_UNROLL) {
                    minMM = 2 * GEMM_UNROLL;
                } else if (minMM > GEMM_UNROLL) {
                    minMM = GEMM_UNROLL;
                }

                PackMatrixA_8(minMM, minK, a + mms * lda + p, lda, packedA + minK * (mms - i));
                
                Kernel_8x8(minMM, minN, minK, packedA + minK * (mms - i), packedB, c + mms * ldc, ldc);
            }

            for (int ns = minN; ns < n; ns += minN) {
                minN = n - ns;
                if (minN > 2 * GEMM_N) {
                    minN = GEMM_N;
                } else if (minN > GEMM_N) {
                    minN = (minN / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                }
                PackMatrixB_8(minK, minN, b + ns + ldb * p, ldb, packedB);
                
                Kernel_8x8(minM, minN, minK, packedA, packedB, c + i * ldc + ns, ldc);
            }
        }
    }

    free(packedA);
    free(packedB);
}
#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define GEMM_M 256
#define GEMM_N 256
#define GEMM_K 256


void Kernel_4x4(int m, int n, int k, float *sa, float *sb, float *sc, int ldc)
{
    
    float32x4_t a_vreg, b_vreg;

    float *bptr = sb;
    float *aptr = sa;
    float *cptr = sc;

    for (int i = 0; i < m; i += 4) {
        cptr = sc + ldc * i;
        for (int j = 0; j < n; j += 4) {

            __builtin_prefetch(aptr, 0, 3);
            __builtin_prefetch(bptr, 0, 3);

            float32x4_t c0_vreg{0}, c1_vreg{0}, c2_vreg{0}, c3_vreg{0};

            for (int p = 0; p < k; p += 4) {
                a_vreg = vld1q_f32(aptr);
                b_vreg = vld1q_f32(bptr);

                c0_vreg = vmlaq_laneq_f32(c0_vreg, b_vreg, a_vreg, 0);
                c1_vreg = vmlaq_laneq_f32(c1_vreg, b_vreg, a_vreg, 1);
                c2_vreg = vmlaq_laneq_f32(c2_vreg, b_vreg, a_vreg, 2);
                c3_vreg = vmlaq_laneq_f32(c3_vreg, b_vreg, a_vreg, 3);

                a_vreg = vld1q_f32(aptr + 4);
                b_vreg = vld1q_f32(bptr + 4);

                c0_vreg = vmlaq_laneq_f32(c0_vreg, b_vreg, a_vreg, 0);
                c1_vreg = vmlaq_laneq_f32(c1_vreg, b_vreg, a_vreg, 1);
                c2_vreg = vmlaq_laneq_f32(c2_vreg, b_vreg, a_vreg, 2);
                c3_vreg = vmlaq_laneq_f32(c3_vreg, b_vreg, a_vreg, 3);

                a_vreg = vld1q_f32(aptr + 8);
                b_vreg = vld1q_f32(bptr + 8);

                c0_vreg = vmlaq_laneq_f32(c0_vreg, b_vreg, a_vreg, 0);
                c1_vreg = vmlaq_laneq_f32(c1_vreg, b_vreg, a_vreg, 1);
                c2_vreg = vmlaq_laneq_f32(c2_vreg, b_vreg, a_vreg, 2);
                c3_vreg = vmlaq_laneq_f32(c3_vreg, b_vreg, a_vreg, 3);

                a_vreg = vld1q_f32(aptr + 12);
                b_vreg = vld1q_f32(bptr + 12);

                c0_vreg = vmlaq_laneq_f32(c0_vreg, b_vreg, a_vreg, 0);
                c1_vreg = vmlaq_laneq_f32(c1_vreg, b_vreg, a_vreg, 1);
                c2_vreg = vmlaq_laneq_f32(c2_vreg, b_vreg, a_vreg, 2);
                c3_vreg = vmlaq_laneq_f32(c3_vreg, b_vreg, a_vreg, 3);

                aptr += 16;
                bptr += 16;
            }

            c0_vreg = vaddq_f32(c0_vreg, vld1q_f32(cptr));
            c1_vreg = vaddq_f32(c1_vreg, vld1q_f32(cptr + ldc));
            c2_vreg = vaddq_f32(c2_vreg, vld1q_f32(cptr + 2 * ldc));
            c3_vreg = vaddq_f32(c3_vreg, vld1q_f32(cptr + 3 * ldc));

            vst1q_f32(cptr, c0_vreg);
            vst1q_f32(cptr + ldc, c1_vreg);
            vst1q_f32(cptr + 2 * ldc, c2_vreg);
            vst1q_f32(cptr + 3 * ldc, c3_vreg);

            cptr += 4;
            aptr -= 4 * k;
        }

        bptr = sb;
        aptr += 4 * k;
    }
}



void PackMatrixA_4(int m, int k, float *a, int lda, float *a_to)
{ 
    float *a0_pntr = a;
    float *a1_pntr = a + k;
    float *a2_pntr = a + 2 * k;
    float *a3_pntr = a + 3 * k;

    int ik = (k >> 2);

    for (int i = 0; i < m; i += 4) {

        for (int j = 0; j < ik; j++) {
            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;

            *a_to++ = *a0_pntr++;
            *a_to++ = *a1_pntr++;
            *a_to++ = *a2_pntr++;
            *a_to++ = *a3_pntr++;
        }
        a0_pntr += 3 * k;
        a1_pntr += 3 * k;
        a2_pntr += 3 * k;
        a3_pntr += 3 * k;
    }
}

void PackMatrixB_4(int k, int n, float *b, int ldb, float *b_to) 
{
    float *btmp0, *btmp1, *btmp2, *btmp3;
    int in = (n >> 2);
    int ik = (k >> 2);

    for (int i = 0; i < in; i++) {
        btmp0 = b + (i << 2);
        for (int j = 0; j < ik; j++) {
            btmp1 = btmp0 + n;
            btmp2 = btmp0 + 2 * n;
            btmp3 = btmp0 + 3 * n;

            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;
            *b_to++ = *btmp0++;

            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;
            *b_to++ = *btmp1++;

            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;
            *b_to++ = *btmp2++;

            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;
            *b_to++ = *btmp3++;

            btmp0 += (4 * n - 4);
        }
    }
}



void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int pb, ib;

    float *packedA = (float*)aligned_alloc(64, m * k * sizeof(float));
    float *packedB = (float*)aligned_alloc(64, k * n * sizeof(float));

    int minM, minN, minK;

    for (int i = 0; i < m; i += minM) {
        minM = std::min(GEMM_M, m - i);

        for (int p = 0; p < k; p += minK) {
            minK = std::min(GEMM_K, k - p);
        

            minN = std::min(n, GEMM_N);
            PackMatrixB_4(minK, minN, &B(0, 0), ldb, packedB);

            int minMM = 4;

            for (int mms = i; mms < i + minM; mms += minMM) {
                PackMatrixA_4(minMM, minK, a + mms * lda + p, lda, packedA + minK * (i - mms));

                Kernel_4x4(minMM, minN, minK, packedA + minK * (i - mms), packedB, c + mms * ldc, ldc);
            }

            for (int ns = minN; ns < n; ns += minN) {
                minN = std::min(n - ns, GEMM_N);
                PackMatrixB_4(minK, minN, b + ns + ldb * p, ldb, packedB);

                Kernel_4x4(minM, minN, minK, packedA, packedB, c + i * ldc + ns, ldc);
            }
        }
    }

    free(packedA);
    free(packedB);

}
#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define GEMM_M 1024
#define GEMM_N 256
#define GEMM_K 256
#define GEMM_UNROLL_N 4
#define GEMM_UNROLL_M 4

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
    int i, j;

    float* a_offset, * a_offset1, * a_offset2, * a_offset3, * a_offset4;
    float* b_offset;
    float  ctemp1, ctemp2, ctemp3, ctemp4;
    float  ctemp5, ctemp6, ctemp7, ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float  ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = a;
    b_offset = a_to;

    j = (m >> 2);
    if (j > 0) 
    {
        do 
        {
            a_offset1 = a_offset;
            a_offset2 = a_offset1 + lda;
            a_offset3 = a_offset2 + lda;
            a_offset4 = a_offset3 + lda;
            a_offset += 4 * lda;

            i = (k >> 2);
            if (i > 0) 
            {
                do 
                {
                    ctemp1 = *(a_offset1 + 0);
                    ctemp2 = *(a_offset1 + 1);
                    ctemp3 = *(a_offset1 + 2);
                    ctemp4 = *(a_offset1 + 3);

                    ctemp5 = *(a_offset2 + 0);
                    ctemp6 = *(a_offset2 + 1);
                    ctemp7 = *(a_offset2 + 2);
                    ctemp8 = *(a_offset2 + 3);

                    ctemp9 = *(a_offset3 + 0);
                    ctemp10 = *(a_offset3 + 1);
                    ctemp11 = *(a_offset3 + 2);
                    ctemp12 = *(a_offset3 + 3);

                    ctemp13 = *(a_offset4 + 0);
                    ctemp14 = *(a_offset4 + 1);
                    ctemp15 = *(a_offset4 + 2);
                    ctemp16 = *(a_offset4 + 3);

                    *(b_offset + 0) = ctemp1;
                    *(b_offset + 1) = ctemp5;
                    *(b_offset + 2) = ctemp9;
                    *(b_offset + 3) = ctemp13;

                    *(b_offset + 4) = ctemp2;
                    *(b_offset + 5) = ctemp6;
                    *(b_offset + 6) = ctemp10;
                    *(b_offset + 7) = ctemp14;

                    *(b_offset + 8) = ctemp3;
                    *(b_offset + 9) = ctemp7;
                    *(b_offset + 10) = ctemp11;
                    *(b_offset + 11) = ctemp15;

                    *(b_offset + 12) = ctemp4;
                    *(b_offset + 13) = ctemp8;
                    *(b_offset + 14) = ctemp12;
                    *(b_offset + 15) = ctemp16;

                    a_offset1 += 4;
                    a_offset2 += 4;
                    a_offset3 += 4;
                    a_offset4 += 4;

                    b_offset += 16;
                    i--;
                } while (i > 0);
            }

            j--;
        } while (j > 0);
    } /* end of if(j > 0) */

    return 0;
}

void PackMatrixB_4(int k, int n, float *b, int ldb, float *b_to) 
{
    int i, j;

	float* a_offset, * a_offset1, * a_offset2, * a_offset3, * a_offset4;
	float* b_offset, * b_offset1, * b_offset2, * b_offset3;
	float  ctemp1, ctemp2, ctemp3, ctemp4;
	float  ctemp5, ctemp6, ctemp7, ctemp8;
	float  ctemp9, ctemp10, ctemp11, ctemp12;
	float ctemp13, ctemp14, ctemp15, ctemp16;

	a_offset = b;
	b_offset = b_to;

	b_offset2 = b + k * n;
	b_offset3 = b + k * n;

	j = (k >> 2);
	if (j > 0) 
    {
		do 
        {
			a_offset1 = a_offset;
			a_offset2 = a_offset1 + ldb;
			a_offset3 = a_offset2 + ldb;
			a_offset4 = a_offset3 + ldb;
			a_offset += 4 * ldb;

			b_offset1 = b_offset;
			b_offset += 16;

			i = (n >> 2);
			if (i > 0) 
            {
				do 
                {
					ctemp1 = *(a_offset1 + 0);
					ctemp2 = *(a_offset1 + 1);
					ctemp3 = *(a_offset1 + 2);
					ctemp4 = *(a_offset1 + 3);

					ctemp5 = *(a_offset2 + 0);
					ctemp6 = *(a_offset2 + 1);
					ctemp7 = *(a_offset2 + 2);
					ctemp8 = *(a_offset2 + 3);

					ctemp9 = *(a_offset3 + 0);
					ctemp10 = *(a_offset3 + 1);
					ctemp11 = *(a_offset3 + 2);
					ctemp12 = *(a_offset3 + 3);

					ctemp13 = *(a_offset4 + 0);
					ctemp14 = *(a_offset4 + 1);
					ctemp15 = *(a_offset4 + 2);
					ctemp16 = *(a_offset4 + 3);

					a_offset1 += 4;
					a_offset2 += 4;
					a_offset3 += 4;
					a_offset4 += 4;

					*(b_offset1 + 0) = ctemp1;
					*(b_offset1 + 1) = ctemp2;
					*(b_offset1 + 2) = ctemp3;
					*(b_offset1 + 3) = ctemp4;

					*(b_offset1 + 4) = ctemp5;
					*(b_offset1 + 5) = ctemp6;
					*(b_offset1 + 6) = ctemp7;
					*(b_offset1 + 7) = ctemp8;

					*(b_offset1 + 8) = ctemp9;
					*(b_offset1 + 9) = ctemp10;
					*(b_offset1 + 10) = ctemp11;
					*(b_offset1 + 11) = ctemp12;

					*(b_offset1 + 12) = ctemp13;
					*(b_offset1 + 13) = ctemp14;
					*(b_offset1 + 14) = ctemp15;
					*(b_offset1 + 15) = ctemp16;

					b_offset1 += k * 4;
					i--;
				} while (i > 0);
			}
			j--;
		} while (j > 0);
	}
	return 0;
}


void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int l2size = GEMM_N * GEMM_K;
    float *sa = (float*)aligned_alloc(64, m * k * sizeof(float));
    float *sb = (float*)aligned_alloc(64, k * n * sizeof(float));

    int ms, ks, ns, mms;
    int minM, minN, minK, minMM;
    int gemmN;

    for (ms = 0; ms < m; ms += minM) {
        minM = m - ms;
        if (minM > GEMM_M) {
            minM = GEMM_M;
        }

        for (ks = 0; ks < k; ks += minK) {
            minK = k - ks;

            if (minK > 2 * GEMM_K) {
                minK = GEMM_K;
                gemmN = GEMM_N;
            } else if (minK > GEMM_K) {
                minK = (minK / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
                gemmN = ((l2size / minK + GEMM_UNROLL_N - 1)) & ~(GEMM_UNROLL_N - 1);
                while (gemmN * minK > l2size) {
                    gemmN -= GEMM_UNROLL_N;
                }
            } 
            
            minN = n;

            if (minN >= GEMM_N * 2) {
                minN = GEMM_N;
            }
            else {
                if (minN > GEMM_N) {
                    minN = ((minN / 2 + GEMM_UNROLL_N - 1) / GEMM_UNROLL_N) * GEMM_UNROLL_N;
                }
            }
    

            PackMatrixB_4(minK, minN, b + ks * ldb, ldb, sb);

            for (mms = ms; mms < ms + minM; mms += minMM) {
                minMM = ms + minM - mms;
                if (minMM > 3 * GEMM_UNROLL_M) {
                    minMM = 3 * GEMM_UNROLL_M;
                } else if (minMM > 2 * GEMM_UNROLL_M) {
                    minMM = 2 * GEMM_UNROLL_M;
                } else if (minMM > GEMM_UNROLL_M) {
                    minMM = GEMM_UNROLL_M;
                }

                PackMatrixA_4(minMM, minK, a + mms * lda + ks, lda, sa + minK * (mms - ms));
                
                Kernel_4x4(minMM, minN, minK, sa + minK * (mms - ms), sb, c + mms * ldc, ldc);
            }

            for (ns = minN; ns < n; ns += minN) {
                minN = n - ns;
                if (minN > 2 * GEMM_N) {
                    minN = GEMM_N;
                } else if (minN > GEMM_N) {
                    minN = (minN / 2 + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1);
                }
                PackMatrixB_4(minK, minN, b + ns + ldb * ks, ldb, sb);
                
                Kernel_4x4(minM, minN, minK, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }

    free(sa);
    free(sb);

}
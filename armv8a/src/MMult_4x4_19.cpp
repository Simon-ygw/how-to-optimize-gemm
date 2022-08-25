#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define GEMM_M 256
#define GEMM_N 256
#define GEMM_K 1024
#define GEMM_UNROLL_N 4
#define GEMM_UNROLL_M 4

void Kernel_4x4(int m, int n, int k, float *sa, float *sb, float *sc, int ldc)
{
    
    float32x4_t a_vreg, b_vreg;

    float *bptr = sb;
    float *aptr = sa;
    float *cptr = sc;
    int ldc_offset = (ldc << 2);
    int ldc_offset_2 = (ldc_offset << 1);
    int ldc_offset_3 = ldc_offset_2 + ldc_offset;
    for (int i = 0; i < m; i += 4) {
        cptr = sc + ldc * i;
        __builtin_prefetch(cptr, 0, 3);
        for (int j = 0; j < n; j += 4) {
            asm volatile(
                "   fmov s15, wzr                     \n"
                "   fmov s16, s15                      \n"
                "   fmov s17, s6                      \n"
                "   fmov s18, s17                      \n"
                "   asr x8,%4,2                      \n"
                "run:                                \n"
                //"   prfm pldl1keep, [%0, #256]       \n"
                //"   prfm pldl1keep, [%1, #256]       \n"
                "   ldp  q13, q24, [%0], #32         \n"  
                "   ldp  q14, q20, [%1], #32         \n"
                "   ldp  q25, q26, [%0], #32         \n"
                "   ldp  q21, q22, [%1], #32         \n"
                "   fmla v15.4s, v14.4s, v13.s[0]    \n"
                "   fmla v16.4s, v14.4s, v13.s[1]    \n"        
                "   fmla v17.4s, v14.4s, v13.s[2]    \n"
                "   fmla v18.4s, v14.4s, v13.s[3]    \n"    

                "   fmla v15.4s, v20.4s, v24.s[0]    \n"
                "   fmla v16.4s, v20.4s, v24.s[1]    \n"
                "   fmla v17.4s, v20.4s, v24.s[2]    \n"
                "   fmla v18.4s, v20.4s, v24.s[3]    \n"

                "   fmla v15.4s, v21.4s, v25.s[0]    \n"
                "   fmla v16.4s, v21.4s, v25.s[1]    \n"
                "   fmla v17.4s, v21.4s, v25.s[2]    \n"
                "   fmla v18.4s, v21.4s, v25.s[3]    \n"

                "   subs x8, x8, #1                  \n"            
                "   fmla v15.4s, v22.4s, v26.s[0]    \n"
                "   fmla v16.4s, v22.4s, v26.s[1]    \n"
                "   fmla v17.4s, v22.4s, v26.s[2]    \n"
                "   fmla v18.4s, v22.4s, v26.s[3]    \n"
                "   bne run                          \n"
                "   add  x13, %2, %3                 \n"
                "   add  x14, %2, %5                 \n"
                "   add  x15, %2, %6                 \n"      
                "   ld1  {v9.4s}, [%2]               \n"
                "   ld1  {v10.4s}, [x13]             \n"
                "   ld1  {v11.4s}, [x14]             \n"
                "   ld1  {v12.4s}, [x15]             \n"
                "   fadd v9.4s, v9.4s, v15.4s        \n"     
                "   fadd v10.4s, v10.4s, v16.4s      \n" 
                "   fadd v11.4s, v11.4s, v17.4s      \n"    
                "   fadd v12.4s, v12.4s, v18.4s      \n"
                "   st1  {v9.4s}, [%2]               \n"
                "   st1  {v10.4s}, [x13]             \n"
                "   st1  {v11.4s}, [x14]             \n"
                "   st1  {v12.4s}, [x15]             \n"   
                : "=r"(aptr), "=r"(bptr), "=r"(cptr), "=r"(ldc_offset), "=r"(k), "=r"(ldc_offset_2), "=r"(ldc_offset_3)
                : "0"(aptr), "1"(bptr), "2"(cptr), "3"(ldc_offset), "4"(k), "5"(ldc_offset_2), "6"(ldc_offset_3)
                : "memory", "cc", "x8", "x9", "x10", "x13", "x14", "x15", "v0", "v1", "v13", "v14","v15", "v16",
                    "v17","v18", "v9","v10","v11","v12", "v20", "v21", "v22", "v23", "v24", "v25", "v26");

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

    for (ks = 0; ks < k; ks += GEMM_K) {
        minK = std::min(k - ks, GEMM_K);
        for (ms = 0; ms < m; ms += minM) {
            minM = std::min(m - ms, GEMM_M);
            PackMatrixA_4(minM, minK, a + ms * lda + ks, lda, sa);

            for (ns = 0; ns < n; ns += minN) {
                minN = std::min(n - ns, GEMM_N);
                PackMatrixB_4(minK, minN, b + ks * ldb, ldb, sb);

                Kernel_4x4(minM, minN, minK, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }
/*
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
    
            if (ms == 0)
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
                if (ms == 0)
                    PackMatrixB_4(minK, minN, b + ns + ldb * ks, ldb, sb);
                
                Kernel_4x4(minM, minN, minK, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }
    */

    free(sa);
    free(sb);

}
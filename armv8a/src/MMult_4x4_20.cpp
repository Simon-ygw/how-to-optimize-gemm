#include <stdlib.h>
#include <algorithm>
#include <arm_neon.h>

#define GEMM_M 1024
#define GEMM_N 256
#define GEMM_K 384
#define GEMM_UNROLL_N 16
#define GEMM_UNROLL_M 4

void Gemm_Kernel_4x4(int K, float *A, float *B, float *C, int LDC)
{
    LDC = (LDC << 2);
    asm volatile(
        "   fmov s15, wzr                    \n"
        "   fmov s16, s15                    \n"
        "   fmov s17, s6                     \n"
        "   fmov s18, s17                    \n"
        "   asr x8,%4,2                      \n"
        "   add  x13, %2, %3                 \n"
        "   add  x14, x13, %3                \n"
        "   add  x15, x14, %3                \n"    
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
        : "=r"(A), "=r"(B), "=r"(C), "=r"(LDC), "=r"(K)
        : "0"(A), "1"(B), "2"(C), "3"(LDC), "4"(K)
        : "memory", "cc", "x8", "x9", "x10", "x13", "x14", "x15", "v0", "v1", "v13", "v14","v15", "v16",
            "v17","v18", "v9","v10","v11","v12", "v20", "v21", "v22", "v23", "v24", "v25", "v26");
}

void Gemm_Kernel_4x8(int K, float *A, float *B, float *C, int LDC) {
    //LDC = (LDC << 2);
    float32x4_t vc00{0}, vc01{0}, 
                vc10{0}, vc11{0},
                vc20{0}, vc21{0},
                vc30{0}, vc31{0};
    for (int i = 0; i < K; i += 4) {
        float32x4_t va0 = vld1q_f32(A);
        float32x4_t vb0 = vld1q_f32(B);
        float32x4_t vb1 = vld1q_f32(B + 4);

        vc00 = vmlaq_laneq_f32(vc00, vb0, va0, 0);
        vc01 = vmlaq_laneq_f32(vc01, vb1, va0, 0);

        vc10 = vmlaq_laneq_f32(vc10, vb0, va0, 1);
        vc11 = vmlaq_laneq_f32(vc11, vb1, va0, 1);

        vc20 = vmlaq_laneq_f32(vc20, vb0, va0, 2);
        vc21 = vmlaq_laneq_f32(vc21, vb1, va0, 2);

        vc30 = vmlaq_laneq_f32(vc30, vb0, va0, 3);
        vc31 = vmlaq_laneq_f32(vc31, vb1, va0, 3);

        va0 = vld1q_f32(A + 4);
        vb0 = vld1q_f32(B + 8);
        vb1 = vld1q_f32(B + 12);

        vc00 = vmlaq_laneq_f32(vc00, vb0, va0, 0);
        vc01 = vmlaq_laneq_f32(vc01, vb1, va0, 0);

        vc10 = vmlaq_laneq_f32(vc10, vb0, va0, 1);
        vc11 = vmlaq_laneq_f32(vc11, vb1, va0, 1);

        vc20 = vmlaq_laneq_f32(vc20, vb0, va0, 2);
        vc21 = vmlaq_laneq_f32(vc21, vb1, va0, 2);

        vc30 = vmlaq_laneq_f32(vc30, vb0, va0, 3);
        vc31 = vmlaq_laneq_f32(vc31, vb1, va0, 3);

        va0 = vld1q_f32(A + 8);
        vb0 = vld1q_f32(B + 16);
        vb1 = vld1q_f32(B + 20);

        vc00 = vmlaq_laneq_f32(vc00, vb0, va0, 0);
        vc01 = vmlaq_laneq_f32(vc01, vb1, va0, 0);

        vc10 = vmlaq_laneq_f32(vc10, vb0, va0, 1);
        vc11 = vmlaq_laneq_f32(vc11, vb1, va0, 1);

        vc20 = vmlaq_laneq_f32(vc20, vb0, va0, 2);
        vc21 = vmlaq_laneq_f32(vc21, vb1, va0, 2);

        vc30 = vmlaq_laneq_f32(vc30, vb0, va0, 3);
        vc31 = vmlaq_laneq_f32(vc31, vb1, va0, 3);

        va0 = vld1q_f32(A + 12);
        vb0 = vld1q_f32(B + 24);
        vb1 = vld1q_f32(B + 28);

        vc00 = vmlaq_laneq_f32(vc00, vb0, va0, 0);
        vc01 = vmlaq_laneq_f32(vc01, vb1, va0, 0);

        vc10 = vmlaq_laneq_f32(vc10, vb0, va0, 1);
        vc11 = vmlaq_laneq_f32(vc11, vb1, va0, 1);

        vc20 = vmlaq_laneq_f32(vc20, vb0, va0, 2);
        vc21 = vmlaq_laneq_f32(vc21, vb1, va0, 2);

        vc30 = vmlaq_laneq_f32(vc30, vb0, va0, 3);
        vc31 = vmlaq_laneq_f32(vc31, vb1, va0, 3);

        A += 16;
        B += 32;
    }

    vc00 = vaddq_f32(vc00, vld1q_f32(C));
    vc01 = vaddq_f32(vc01, vld1q_f32(C + 4));
    vc10 = vaddq_f32(vc10, vld1q_f32(C + LDC));
    vc11 = vaddq_f32(vc11, vld1q_f32(C + LDC + 4));
    vc20 = vaddq_f32(vc20, vld1q_f32(C + 2 * LDC));
    vc21 = vaddq_f32(vc21, vld1q_f32(C + 2 * LDC + 4));
    vc30 = vaddq_f32(vc30, vld1q_f32(C + 3 * LDC));
    vc31 = vaddq_f32(vc31, vld1q_f32(C + 3 * LDC + 4));

    vst1q_f32(C, vc00);
    vst1q_f32(C + 4, vc01);
    vst1q_f32(C + LDC, vc10);
    vst1q_f32(C + LDC + 4, vc11);
    vst1q_f32(C + 2 * LDC, vc20);
    vst1q_f32(C + 2 * LDC + 4, vc21);
    vst1q_f32(C + 3 * LDC, vc30);
    vst1q_f32(C + 3 * LDC + 4, vc31);  
}

void Gemm_Kernel_4x16(int K, float *A, float *B, float *C, int LDC) {

    LDC = (LDC << 2);
    asm volatile(
        "   mov  x13, %2                     \n"
        "   ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], %3 \n"
        "   ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], %3 \n"
        "   ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], %3 \n"
        "   ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], %3 \n"
        "   asr x8,%4,2                      \n"  
        "run1:                               \n"
        "   ld1  {v0.4s, v1.4s}, [%0], #32   \n"
        "   ld1  {v2.4s, v3.4s}, [%1], #32   \n"
        "   ld1  {v4.4s, v5.4s}, [%1], #32   \n"

        "   fmla v16.4s, v2.4s, v0.s[0]      \n"
        "   fmla v20.4s, v2.4s, v0.s[1]      \n"
        "   fmla v24.4s, v2.4s, v0.s[2]      \n"
        "   fmla v28.4s, v2.4s, v0.s[3]      \n"
        "   fmla v17.4s, v3.4s, v0.s[0]      \n"
        "   fmla v21.4s, v3.4s, v0.s[1]      \n"
        "   fmla v25.4s, v3.4s, v0.s[2]      \n"
        "   fmla v29.4s, v3.4s, v0.s[3]      \n"
        "   ld1  {v2.4s, v3.4s}, [%1], #32   \n"

        "   fmla v18.4s, v4.4s, v0.s[0]      \n"
        "   fmla v22.4s, v4.4s, v0.s[1]      \n"
        "   fmla v26.4s, v4.4s, v0.s[2]      \n"
        "   fmla v30.4s, v4.4s, v0.s[3]      \n"
        "   fmla v19.4s, v5.4s, v0.s[0]      \n"      
        "   fmla v23.4s, v5.4s, v0.s[1]      \n"    
        "   fmla v27.4s, v5.4s, v0.s[2]      \n"  
        "   fmla v31.4s, v5.4s, v0.s[3]      \n" 
        "   subs x8, x8, #1                  \n"    

        "   ld1  {v4.4s, v5.4s}, [%1], #32   \n" 

        "   fmla v16.4s, v2.4s, v1.s[0]      \n"
        "   fmla v20.4s, v2.4s, v1.s[1]      \n"
        "   fmla v24.4s, v2.4s, v1.s[2]      \n"
        "   fmla v28.4s, v2.4s, v1.s[3]      \n"
        "   fmla v17.4s, v3.4s, v1.s[0]      \n"
        "   fmla v21.4s, v3.4s, v1.s[1]      \n"
        "   fmla v25.4s, v3.4s, v1.s[2]      \n"
        "   fmla v29.4s, v3.4s, v1.s[3]      \n"
        "   ld1  {v2.4s, v3.4s}, [%1], #32   \n"

        "   fmla v18.4s, v4.4s, v1.s[0]      \n"
        "   fmla v22.4s, v4.4s, v1.s[1]      \n"
        "   fmla v26.4s, v4.4s, v1.s[2]      \n"
        "   fmla v30.4s, v4.4s, v1.s[3]      \n"
        "   fmla v19.4s, v5.4s, v1.s[0]      \n"      
        "   fmla v23.4s, v5.4s, v1.s[1]      \n"    
        "   fmla v27.4s, v5.4s, v1.s[2]      \n"  
        "   fmla v31.4s, v5.4s, v1.s[3]      \n"     

        "   ld1  {v0.4s, v1.4s}, [%0], #32   \n" 
        "   ld1  {v4.4s, v5.4s}, [%1], #32   \n"

        "   fmla v16.4s, v2.4s, v0.s[0]      \n"
        "   fmla v20.4s, v2.4s, v0.s[1]      \n"
        "   fmla v24.4s, v2.4s, v0.s[2]      \n"
        "   fmla v28.4s, v2.4s, v0.s[3]      \n"
        "   fmla v17.4s, v3.4s, v0.s[0]      \n"
        "   fmla v21.4s, v3.4s, v0.s[1]      \n"
        "   fmla v25.4s, v3.4s, v0.s[2]      \n"
        "   fmla v29.4s, v3.4s, v0.s[3]      \n"
        "   ld1  {v2.4s, v3.4s}, [%1], #32   \n"

        "   fmla v18.4s, v4.4s, v0.s[0]      \n"
        "   fmla v22.4s, v4.4s, v0.s[1]      \n"
        "   fmla v26.4s, v4.4s, v0.s[2]      \n"
        "   fmla v30.4s, v4.4s, v0.s[3]      \n"
        "   fmla v19.4s, v5.4s, v0.s[0]      \n"      
        "   fmla v23.4s, v5.4s, v0.s[1]      \n"    
        "   fmla v27.4s, v5.4s, v0.s[2]      \n"  
        "   fmla v31.4s, v5.4s, v0.s[3]      \n" 

        "   ld1  {v4.4s, v5.4s}, [%1], #32   \n"

        "   fmla v16.4s, v2.4s, v1.s[0]      \n"
        "   fmla v20.4s, v2.4s, v1.s[1]      \n"
        "   fmla v24.4s, v2.4s, v1.s[2]      \n"
        "   fmla v28.4s, v2.4s, v1.s[3]      \n"
        "   fmla v17.4s, v3.4s, v1.s[0]      \n"
        "   fmla v21.4s, v3.4s, v1.s[1]      \n"
        "   fmla v25.4s, v3.4s, v1.s[2]      \n"
        "   fmla v29.4s, v3.4s, v1.s[3]      \n"

        "   fmla v18.4s, v4.4s, v1.s[0]      \n"
        "   fmla v22.4s, v4.4s, v1.s[1]      \n"
        "   fmla v26.4s, v4.4s, v1.s[2]      \n"
        "   fmla v30.4s, v4.4s, v1.s[3]      \n"
        "   fmla v19.4s, v5.4s, v1.s[0]      \n"      
        "   fmla v23.4s, v5.4s, v1.s[1]      \n"    
        "   fmla v27.4s, v5.4s, v1.s[2]      \n"  
        "   fmla v31.4s, v5.4s, v1.s[3]      \n"     
        
        "   bne run1                          \n"   
        "   st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x13], %3 \n"
        "   st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x13], %3 \n"
        "   st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x13], %3 \n"
        "   st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x13], %3 \n"
        : "=r"(A), "=r"(B), "=r"(C), "=r"(LDC), "=r"(K)
        : "0"(A), "1"(B), "2"(C), "3"(LDC), "4"(K)
        : "memory", "cc", "x8","x13", "v0", "v1", "v2", "v3", "v4", "v5", "v13", "v14","v15", "v16",
            "v17","v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");  
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
}

void PackMatrixB_16(int m, int n, float *b, int ldb, float *b_to) 
{
    int i, j;

    float *aoffset;
    float *aoffset1, *aoffset2;
    float *boffset;

    float ctemp01, ctemp02, ctemp03, ctemp04;
    float ctemp05, ctemp06, ctemp07, ctemp08;
    float ctemp09, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;
    float ctemp17, ctemp18, ctemp19, ctemp20;
    float ctemp21, ctemp22, ctemp23, ctemp24;
    float ctemp25, ctemp26, ctemp27, ctemp28;
    float ctemp29, ctemp30, ctemp31, ctemp32;

    aoffset   = b;
    boffset   = b_to;

    j = (n >> 4);
    if (j > 0){
        do{
            aoffset1  = aoffset;
            aoffset2  = aoffset + ldb;
            aoffset += 16;

            i = (m >> 1);
            if (i > 0){
                do{
                    ctemp01 = *(aoffset1 +  0);
                    ctemp02 = *(aoffset1 +  1);
                    ctemp03 = *(aoffset1 +  2);
                    ctemp04 = *(aoffset1 +  3);
                    ctemp05 = *(aoffset1 +  4);
                    ctemp06 = *(aoffset1 +  5);
                    ctemp07 = *(aoffset1 +  6);
                    ctemp08 = *(aoffset1 +  7);
                    ctemp09 = *(aoffset1 +  8);
                    ctemp10 = *(aoffset1 +  9);
                    ctemp11 = *(aoffset1 + 10);
                    ctemp12 = *(aoffset1 + 11);
                    ctemp13 = *(aoffset1 + 12);
                    ctemp14 = *(aoffset1 + 13);
                    ctemp15 = *(aoffset1 + 14);
                    ctemp16 = *(aoffset1 + 15);

                    ctemp17 = *(aoffset2 +  0);
                    ctemp18 = *(aoffset2 +  1);
                    ctemp19 = *(aoffset2 +  2);
                    ctemp20 = *(aoffset2 +  3);
                    ctemp21 = *(aoffset2 +  4);
                    ctemp22 = *(aoffset2 +  5);
                    ctemp23 = *(aoffset2 +  6);
                    ctemp24 = *(aoffset2 +  7);
                    ctemp25 = *(aoffset2 +  8);
                    ctemp26 = *(aoffset2 +  9);
                    ctemp27 = *(aoffset2 + 10);
                    ctemp28 = *(aoffset2 + 11);
                    ctemp29 = *(aoffset2 + 12);
                    ctemp30 = *(aoffset2 + 13);
                    ctemp31 = *(aoffset2 + 14);
                    ctemp32 = *(aoffset2 + 15);

                    *(boffset +  0) = ctemp01;
                    *(boffset +  1) = ctemp02;
                    *(boffset +  2) = ctemp03;
                    *(boffset +  3) = ctemp04;
                    *(boffset +  4) = ctemp05;
                    *(boffset +  5) = ctemp06;
                    *(boffset +  6) = ctemp07;
                    *(boffset +  7) = ctemp08;

                    *(boffset +  8) = ctemp09;
                    *(boffset +  9) = ctemp10;
                    *(boffset + 10) = ctemp11;
                    *(boffset + 11) = ctemp12;
                    *(boffset + 12) = ctemp13;
                    *(boffset + 13) = ctemp14;
                    *(boffset + 14) = ctemp15;
                    *(boffset + 15) = ctemp16;

                    *(boffset + 16) = ctemp17;
                    *(boffset + 17) = ctemp18;
                    *(boffset + 18) = ctemp19;
                    *(boffset + 19) = ctemp20;
                    *(boffset + 20) = ctemp21;
                    *(boffset + 21) = ctemp22;
                    *(boffset + 22) = ctemp23;
                    *(boffset + 23) = ctemp24;

                    *(boffset + 24) = ctemp25;
                    *(boffset + 25) = ctemp26;
                    *(boffset + 26) = ctemp27;
                    *(boffset + 27) = ctemp28;
                    *(boffset + 28) = ctemp29;
                    *(boffset + 29) = ctemp30;
                    *(boffset + 30) = ctemp31;
                    *(boffset + 31) = ctemp32;

                    aoffset1 +=  2 * ldb;
                    aoffset2 +=  2 * ldb;
                    boffset   += 32;

                    i --;
                }while(i > 0);
            }

            if (m & 1){
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);
                ctemp05 = *(aoffset1 +  4);
                ctemp06 = *(aoffset1 +  5);
                ctemp07 = *(aoffset1 +  6);
                ctemp08 = *(aoffset1 +  7);
                ctemp09 = *(aoffset1 +  8);
                ctemp10 = *(aoffset1 +  9);
                ctemp11 = *(aoffset1 + 10);
                ctemp12 = *(aoffset1 + 11);
                ctemp13 = *(aoffset1 + 12);
                ctemp14 = *(aoffset1 + 13);
                ctemp15 = *(aoffset1 + 14);
                ctemp16 = *(aoffset1 + 15);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;

                *(boffset +  8) = ctemp09;
                *(boffset +  9) = ctemp10;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp12;
                *(boffset + 12) = ctemp13;
                *(boffset + 13) = ctemp14;
                *(boffset + 14) = ctemp15;
                *(boffset + 15) = ctemp16;

                boffset   += 16;
            }

        j--;
        }while(j > 0);
    } /* end of if(j > 0) */

    if (n & 8){
        aoffset1  = aoffset;
        aoffset2  = aoffset + ldb;
        aoffset += 8;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);
                ctemp05 = *(aoffset1 +  4);
                ctemp06 = *(aoffset1 +  5);
                ctemp07 = *(aoffset1 +  6);
                ctemp08 = *(aoffset1 +  7);

                ctemp09 = *(aoffset2 +  0);
                ctemp10 = *(aoffset2 +  1);
                ctemp11 = *(aoffset2 +  2);
                ctemp12 = *(aoffset2 +  3);
                ctemp13 = *(aoffset2 +  4);
                ctemp14 = *(aoffset2 +  5);
                ctemp15 = *(aoffset2 +  6);
                ctemp16 = *(aoffset2 +  7);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;

                *(boffset +  8) = ctemp09;
                *(boffset +  9) = ctemp10;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp12;
                *(boffset + 12) = ctemp13;
                *(boffset + 13) = ctemp14;
                *(boffset + 14) = ctemp15;
                *(boffset + 15) = ctemp16;

                aoffset1 +=  2 * ldb;
                aoffset2 +=  2 * ldb;
                boffset   += 16;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);
            ctemp03 = *(aoffset1 +  2);
            ctemp04 = *(aoffset1 +  3);
            ctemp05 = *(aoffset1 +  4);
            ctemp06 = *(aoffset1 +  5);
            ctemp07 = *(aoffset1 +  6);
            ctemp08 = *(aoffset1 +  7);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            *(boffset +  2) = ctemp03;
            *(boffset +  3) = ctemp04;
            *(boffset +  4) = ctemp05;
            *(boffset +  5) = ctemp06;
            *(boffset +  6) = ctemp07;
            *(boffset +  7) = ctemp08;

            boffset   += 8;
        }
    }

    if (n & 4){
        aoffset1  = aoffset;
        aoffset2  = aoffset + ldb;
        aoffset += 4;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset1 +  2);
                ctemp04 = *(aoffset1 +  3);

                ctemp05 = *(aoffset2 +  0);
                ctemp06 = *(aoffset2 +  1);
                ctemp07 = *(aoffset2 +  2);
                ctemp08 = *(aoffset2 +  3);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;
                *(boffset +  4) = ctemp05;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp07;
                *(boffset +  7) = ctemp08;

                aoffset1 +=  2 * ldb;
                aoffset2 +=  2 * ldb;
                boffset   += 8;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);
            ctemp03 = *(aoffset1 +  2);
            ctemp04 = *(aoffset1 +  3);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            *(boffset +  2) = ctemp03;
            *(boffset +  3) = ctemp04;

            boffset   += 4;
        }
    }

    if (n & 2){
        aoffset1  = aoffset;
        aoffset2  = aoffset + ldb;
        aoffset += 2;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset1 +  1);
                ctemp03 = *(aoffset2 +  0);
                ctemp04 = *(aoffset2 +  1);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;
                *(boffset +  2) = ctemp03;
                *(boffset +  3) = ctemp04;

                aoffset1 +=  2 * ldb;
                aoffset2 +=  2 * ldb;
                boffset   += 4;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            ctemp02 = *(aoffset1 +  1);

            *(boffset +  0) = ctemp01;
            *(boffset +  1) = ctemp02;
            boffset   += 2;
        }
    }

    if (n & 1){
        aoffset1  = aoffset;
        aoffset2  = aoffset + ldb;

        i = (m >> 1);
        if (i > 0){
            do{
                ctemp01 = *(aoffset1 +  0);
                ctemp02 = *(aoffset2 +  0);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp02;

                aoffset1 +=  2 * ldb;
                aoffset2 +=  2 * ldb;
                boffset   += 2;

                i --;
            }while(i > 0);
        }

        if (m & 1){
            ctemp01 = *(aoffset1 +  0);
            *(boffset +  0) = ctemp01;
            boffset   += 1;
        }
    }
}

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int ms, ns, ks, mms, nns;
    int minM, minN, minK;
    float *sa = (float*)aligned_alloc(64, m * k * sizeof(float));
    float *sb = (float*)aligned_alloc(64, k * n * sizeof(float));

    for (ks = 0; ks < k; ks += minK)
    {
        minK = std::min(k - ks, GEMM_K);

        for (ms = 0; ms < m; ms += minM)
        {
            minM = std::min(m - ms, GEMM_M);
            PackMatrixA_4(minM, minK, a + ms * lda + ks, lda, sa);

            for (ns = 0; ns < n; ns += minN)
            {
                minN = std::min(n - ns, GEMM_N);
                if (ms == 0)
                    PackMatrixB_16(minK, minN, b + ks * ldb + ns, ldb, sb);

                for (mms = ms; mms < ms + minM; mms += GEMM_UNROLL_M) {
                    for (nns = ns; nns < ns + minN - GEMM_UNROLL_N + 1; nns += GEMM_UNROLL_N) {
                        Gemm_Kernel_4x16(minK, sa + minK * (mms - ms), sb + (nns - ns) * minK, c + mms * ldc + nns, ldc);
                    }
                    for (; nns < ns + minN - 7; nns += 8) {
                        Gemm_Kernel_4x8(minK, sa + minK * (mms - ms), sb + (nns - ns) * minK, c + mms * ldc + nns, ldc);
                    } 

                    for (; nns < ns + minN - 3; nns += 4) {
                        Gemm_Kernel_4x4(minK, sa + minK * (mms - ms), sb + (nns - ns) * minK, c + mms * ldc + nns, ldc);
                    }  
                }
            }
        }
    }

    free(sa);
    free(sb);
}

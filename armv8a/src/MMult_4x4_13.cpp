
#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]
#include <algorithm>
#include <arm_neon.h>



void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg;

    c0_vreg = vld1q_f32(&C(0, 0));
    c1_vreg = vld1q_f32(&C(1, 0));
    c2_vreg = vld1q_f32(&C(2, 0));
    c3_vreg = vld1q_f32(&C(3, 0));

    float32x4_t bp_vreg;
    float32x4_t ap_vreg;

    for (int p = 0; p < k; p++) {
        ap_vreg = vld1q_f32(&A(p, 0));
        bp_vreg = vld1q_f32(&(B(p, 0)));

        c0_vreg = vfmaq_f32(c0_vreg, bp_vreg, vdupq_n_f32(vgetq_lane_f32(ap_vreg, 0)));
        c1_vreg = vfmaq_f32(c1_vreg, bp_vreg, vdupq_n_f32(vgetq_lane_f32(ap_vreg, 1)));
        c2_vreg = vfmaq_f32(c2_vreg, bp_vreg, vdupq_n_f32(vgetq_lane_f32(ap_vreg, 2)));
        c3_vreg = vfmaq_f32(c3_vreg, bp_vreg, vdupq_n_f32(vgetq_lane_f32(ap_vreg, 3)));

    }

    vst1q_f32(&C(0, 0), c0_vreg);
    vst1q_f32(&C(1, 0), c1_vreg);
    vst1q_f32(&C(2, 0), c2_vreg);
    vst1q_f32(&C(3, 0), c3_vreg);
}


#define mc 256
#define kc 128

void PackMatrixA(int k, float *a, int lda, float *a_to)
{
    float *a0_pntr = &A(0, 0);
    float *a1_pntr = &A(1, 0);
    float *a2_pntr = &A(2, 0);
    float *a3_pntr = &A(3, 0);
    for (int j = 0; j < k; j++) {
        *a_to++ = *a0_pntr++;
        *a_to++ = *a1_pntr++;
        *a_to++ = *a2_pntr++;
        *a_to++ = *a3_pntr++;
    }
}

void PackMatrixB(int k, float *b, int ldb, float *b_to) 
{
    for (int i = 0; i < k; i++) {
        float *bp_pntr = &B(i, 0);
        
        *b_to++ = *bp_pntr;
        *b_to++ = *(bp_pntr + 1);
        *b_to++ = *(bp_pntr + 2);
        *b_to++ = *(bp_pntr + 3);
    }
}

void InnerKernel(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) 
{
    float packedA[m * k], packedB[k * n];
    for (int i = 0; i < m; i += 4) {
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
		for (int j = 0; j < n; j+=4) {
            if (i == 0)
                PackMatrixB(k, &B(0, j), ldb, &packedB[j * k]);
            AddDot4x4(k, &packedA[i * k], 4, &packedB[j * k], 4, &C(i, j), ldc);

		}
	}
}

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    int pb, ib;
    for (int i = 0; i < m; i += mc) {
        ib = std::min(m - i, mc);
        for (int p = 0; p < k; p += kc) {
            pb = std::min(kc, k - p);
            InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
        }
    }
}
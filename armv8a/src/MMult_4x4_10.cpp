
#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not suppoted")
#endif

void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    float32x4_t c0_vreg, c1_vreg, c2_vreg, c3_vreg;

    c0_vreg = vld1q_f32(&C(0, 0));
    c1_vreg = vld1q_f32(&C(1, 0));
    c2_vreg = vld1q_f32(&C(2, 0));
    c3_vreg = vld1q_f32(&C(3, 0));

    float *a0p_pntr = &A(0, 0);
    float *a1p_pntr = &A(1, 0);
    float *a2p_pntr = &A(2, 0);
    float *a3p_pntr = &A(3, 0);

    float32x4_t bp_vreg;
    float32x4_t a0_vreg, a1_vreg, a2_vreg, a3_vreg;

    for (int p = 0; p < k; p++) {

        bp_vreg = vld1q_f32(&(B(p, 0)));
        a0_vreg = vld1q_dup_f32(a0p_pntr);
        a1_vreg = vld1q_dup_f32(a1p_pntr);
        a2_vreg = vld1q_dup_f32(a2p_pntr);
        a3_vreg = vld1q_dup_f32(a3p_pntr);

        c0_vreg = vfmaq_f32(c0_vreg, bp_vreg, a0_vreg);
        c1_vreg = vfmaq_f32(c1_vreg, bp_vreg, a1_vreg);
        c2_vreg = vfmaq_f32(c2_vreg, bp_vreg, a2_vreg);
        c3_vreg = vfmaq_f32(c3_vreg, bp_vreg, a3_vreg);

        a0p_pntr ++;
        a1p_pntr ++;
        a2p_pntr ++;
        a3p_pntr ++;
    }

    vst1q_f32(&C(0, 0), c0_vreg);
    vst1q_f32(&C(1, 0), c1_vreg);
    vst1q_f32(&C(2, 0), c2_vreg);
    vst1q_f32(&C(3, 0), c3_vreg);
}

void MY_MMult_4x4_10(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
		for (int j = 0; j < n; j+=4) {

            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);

		}
	}
}
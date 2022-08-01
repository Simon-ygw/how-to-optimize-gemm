
#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]


void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    register float c00_reg, c10_reg, c20_reg, c30_reg, 
                    c01_reg, c11_reg, c21_reg, c31_reg,
                    c02_reg, c12_reg, c22_reg, c32_reg,
                    c03_reg, c13_reg, c23_reg, c33_reg, 
                    bp0_reg, bp1_reg, bp2_reg, bp3_reg;

    c00_reg = 0.0;
    c10_reg = 0.0;
    c20_reg = 0.0;
    c30_reg = 0.0;

    c01_reg = 0.0;
    c11_reg = 0.0;
    c21_reg = 0.0;
    c31_reg = 0.0;

    c02_reg = 0.0;
    c12_reg = 0.0;
    c22_reg = 0.0;
    c32_reg = 0.0;

    c03_reg = 0.0;
    c13_reg = 0.0;
    c23_reg = 0.0;
    c33_reg = 0.0;

    float *a0p_pntr = &A(0, 0);
    float *a1p_pntr = &A(1, 0);
    float *a2p_pntr = &A(2, 0);
    float *a3p_pntr = &A(3, 0);

    for (int p = 0; p < k; p+=4) {
        bp0_reg = B(p, 0);
        c00_reg += bp0_reg * *a0p_pntr; 
        c10_reg += bp0_reg * *a1p_pntr; 
        c20_reg += bp0_reg * *a2p_pntr;  
        c30_reg += bp0_reg * *a3p_pntr;  

        bp0_reg = B(p + 1, 0);
        c00_reg += bp0_reg * *(a0p_pntr+1); 
        c10_reg += bp0_reg * *(a1p_pntr+1); 
        c20_reg += bp0_reg * *(a2p_pntr+1);  
        c30_reg += bp0_reg * *(a3p_pntr+1); 

        bp0_reg = B(p + 2, 0);
        c00_reg += bp0_reg * *(a0p_pntr+2); 
        c10_reg += bp0_reg * *(a1p_pntr+2); 
        c20_reg += bp0_reg * *(a2p_pntr+2);  
        c30_reg += bp0_reg * *(a3p_pntr+2); 

        bp0_reg = B(p + 3, 0);
        c00_reg += bp0_reg * *(a0p_pntr+3); 
        c10_reg += bp0_reg * *(a1p_pntr+3); 
        c20_reg += bp0_reg * *(a2p_pntr+3);  
        c30_reg += bp0_reg * *(a3p_pntr+3); 

        bp1_reg = B(p, 1);
        c01_reg += bp1_reg * *a0p_pntr; 
        c11_reg += bp1_reg * *a1p_pntr; 
        c21_reg += bp1_reg * *a2p_pntr;  
        c31_reg += bp1_reg * *a3p_pntr;  

        bp1_reg = B(p + 1, 1);
        c01_reg += bp1_reg * *(a0p_pntr+1); 
        c11_reg += bp1_reg * *(a1p_pntr+1); 
        c21_reg += bp1_reg * *(a2p_pntr+1);  
        c31_reg += bp1_reg * *(a3p_pntr+1); 

        bp1_reg = B(p + 2, 1);
        c01_reg += bp1_reg * *(a0p_pntr+2); 
        c11_reg += bp1_reg * *(a1p_pntr+2); 
        c21_reg += bp1_reg * *(a2p_pntr+2);  
        c31_reg += bp1_reg * *(a3p_pntr+2); 

        bp1_reg = B(p + 3, 1);
        c01_reg += bp1_reg * *(a0p_pntr+3); 
        c11_reg += bp1_reg * *(a1p_pntr+3); 
        c21_reg += bp1_reg * *(a2p_pntr+3);  
        c31_reg += bp1_reg * *(a3p_pntr+3); 

        bp2_reg = B(p, 2);
        c02_reg += bp2_reg * *a0p_pntr; 
        c12_reg += bp2_reg * *a1p_pntr; 
        c22_reg += bp2_reg * *a2p_pntr;  
        c32_reg += bp2_reg * *a3p_pntr;  

        bp2_reg = B(p + 1, 2);
        c02_reg += bp2_reg * *(a0p_pntr+1); 
        c12_reg += bp2_reg * *(a1p_pntr+1); 
        c22_reg += bp2_reg * *(a2p_pntr+1);  
        c32_reg += bp2_reg * *(a3p_pntr+1); 

        bp2_reg = B(p + 2, 2);
        c02_reg += bp2_reg * *(a0p_pntr+2); 
        c12_reg += bp2_reg * *(a1p_pntr+2); 
        c22_reg += bp2_reg * *(a2p_pntr+2);  
        c32_reg += bp2_reg * *(a3p_pntr+2); 

        bp2_reg = B(p + 3, 2);
        c02_reg += bp2_reg * *(a0p_pntr+3); 
        c12_reg += bp2_reg * *(a1p_pntr+3); 
        c22_reg += bp2_reg * *(a2p_pntr+3);  
        c32_reg += bp2_reg * *(a3p_pntr+3); 


        bp3_reg = B(p, 3);
        c03_reg += bp3_reg * *a0p_pntr; 
        c13_reg += bp3_reg * *a1p_pntr; 
        c23_reg += bp3_reg * *a2p_pntr;  
        c33_reg += bp3_reg * *a3p_pntr;  

        bp3_reg = B(p + 1, 3);
        c03_reg += bp3_reg * *(a0p_pntr+1); 
        c13_reg += bp3_reg * *(a1p_pntr+1); 
        c23_reg += bp3_reg * *(a2p_pntr+1);  
        c33_reg += bp3_reg * *(a3p_pntr+1); 

        bp3_reg = B(p + 2, 3);
        c03_reg += bp3_reg * *(a0p_pntr+2); 
        c13_reg += bp3_reg * *(a1p_pntr+2); 
        c23_reg += bp3_reg * *(a2p_pntr+2);  
        c33_reg += bp3_reg * *(a3p_pntr+2); 

        bp3_reg = B(p + 3, 3);
        c03_reg += bp3_reg * *(a0p_pntr+3); 
        c13_reg += bp3_reg * *(a1p_pntr+3); 
        c23_reg += bp3_reg * *(a2p_pntr+3);  
        c33_reg += bp3_reg * *(a3p_pntr+3); 

        a0p_pntr += 4;
        a1p_pntr += 4;
        a2p_pntr += 4;
        a3p_pntr += 4;
    }

    C(0, 0) += c00_reg;
    C(1, 0) += c10_reg;
    C(2, 0) += c20_reg;
    C(3, 0) += c30_reg;

    C(0, 1) += c01_reg;
    C(1, 1) += c11_reg;
    C(2, 1) += c21_reg;
    C(3, 1) += c31_reg;

    C(0, 2) += c02_reg;
    C(1, 2) += c12_reg;
    C(2, 2) += c22_reg;
    C(3, 2) += c32_reg;

    C(0, 3) += c03_reg;
    C(1, 3) += c13_reg;
    C(2, 3) += c23_reg;
    C(3, 3) += c33_reg;
}

void MY_MMult_4x4_9(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < m; i += 4) {
		for (int j = 0; j < n; j+=4) {

            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);

		}
	}
}
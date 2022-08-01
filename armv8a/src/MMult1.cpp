
#define A(i, j)  a[(i) * lda + (j)]
#define B(i, j)  b[(i) * ldb + (j)]
#define C(i, j)  c[(i) * ldc + (j)]

#define Y(i) y[(i) * incy]

void AddDot(int k, float *x, int incy, float *y, float *gamma)
{
    for (int p = 0; p < k; p++) {
        *gamma += x[p] * Y(p);
    }
}

void MY_MMult(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {

            AddDot(k, &A(i, 0), ldb, &B(0, j), &C(i, j));

		}
	}
}
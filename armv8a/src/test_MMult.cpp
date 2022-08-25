#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

class TimeInterval
{
  public:
    TimeInterval() {
      start_ = std::chrono::high_resolution_clock::now();
    }

    double Delta() {
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
      return diff * 1.0e-9;
    }
  private:
    std::chrono::high_resolution_clock::time_point start_;
};

void REF_MMult(int, int, int, float *, int, float *, int, float *, int );
void MY_MMult(int, int, int, float *, int, float *, int, float *, int );

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]


float compare_matrices( int m, int n, float *a, int lda, float *b, int ldb )
{
  int i, j;
  float max_diff = 0.0, diff;

  for ( i=0; j<m; i++ )
    for ( j=0; j<n; j++ ){
      diff = std::abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
  }

  return max_diff;
}

void copy_matrix( int m, int n, float *a, int lda, float *b, int ldb )
{
  int i, j;
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      B( i,j ) = A( i,j );
}

void print_matrix( int m, int n, float *a, int lda )
{
  int i, j;
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ){ 
      printf("%le ", A( i,j ) );
  }
    printf("\n");
  }
  printf("\n");
}

void random_matrix( int m, int n, float *a, int lda )
{
  int i,j;

  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      A( i,j ) = (float)(rand() % 32) / 64.0;
    }
  }
}


int main()
{
  int p, m, n, k, lda, ldb, ldc, rep;

  double dtime, dtime_best, gflops, diff;

  float *a, *b, *c, *cref, *cold;    
  
  printf( "MY_MMult = [\n" );
    
  for ( p=40; p<=800; p+=40 ){
    //p = 40;
    m = p;
    n = p;
    k = p;

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = k;
    ldb = n;
    ldc = n;

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( float * ) aligned_alloc(64, lda * k * sizeof( float ));  
    b = ( float * ) aligned_alloc(64, ldb * n * sizeof( float ));
    c = ( float * ) aligned_alloc(64, ldc * n * sizeof( float ));
    cold = ( float * ) aligned_alloc(64, ldc * n * sizeof( float ));
    cref = ( float * ) aligned_alloc(64, ldc * n * sizeof( float ));


    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, a, lda );
    random_matrix( k, n, b, ldb );
    random_matrix( m, n, cold, ldc );


    copy_matrix( m, n, cold, ldc, cref, ldc );

    /* Run the reference implementation so the answers can be compared */

    REF_MMult( m, n, k, a, lda, b, ldb, cref, ldc );

    //print_matrix(m, n, cref, ldc);

    /* Time the "optimized" implementation */
    for ( rep=0; rep<20; rep++ ){
      copy_matrix( m, n, cold, ldc, c, ldc );

      /* Time your implementation */
      TimeInterval timer;

      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
      
      dtime = timer.Delta();

      if ( rep==0 )
	      dtime_best = dtime;
      else
	      dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    diff = compare_matrices( m, n, c, ldc, cref, ldc );

    //print_matrix(m, n, c, ldc);

    if (diff > 0.5f || diff < 0.5f) {
      //exit(0);
    }

    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( cold );
    free( cref );
  }

  printf( "];\n" );

  exit( 0 );
}


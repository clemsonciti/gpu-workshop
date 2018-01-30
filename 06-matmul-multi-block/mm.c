#include <stdio.h>
#include <cuda.h>

#define N 1024

void mm(double* A, double *B, double* C) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            double Cval = 0;
            for (int k=0; k<N; k++) {
                Cval += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = Cval;
        }
    }
}

int main () {
    
    double *a, *b, *c;   
    double size = sizeof(double) * N*N;

    a = (double *) malloc (size);
    b = (double *) malloc (size);
    c = (double *) malloc (size);

    for( int i = 0; i < N*N; i++ )
    {
      a[i] = (double) ( rand() ) / ( RAND_MAX + 1.0 );
      b[i] = (double) ( rand() ) / ( RAND_MAX + 1.0 );
      c[i] = 0;
    }
    
    for (int i=0; i<5; i++) {
        mm(a, b, c);
    }

    printf("%f\n", c[N * N/2]);

    free(a);
    free(b);
    free(c);

    return 0;
}

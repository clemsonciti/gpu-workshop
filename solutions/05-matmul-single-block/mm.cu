#include <stdio.h>
#include <cuda.h>

#define N 32

__global__ void mm(double *a, double *b, double *c)
{

    /* From: https://wiki.tiker.net/PyCuda/Examples/MatrixmulSimple */

    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    double Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //  to produce Pvalue:
    for (int k = 0; k < N; ++k) {
        double Aelement = a[ty * N + k];
        double Belement = b[k * N + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * N + tx] = Pvalue;
}   

int main () {
    
    double *a, *b, *c;
    double *d_a, *d_b, *d_c;
    double size = sizeof(double) * N*N;
    dim3 grid(1, 1);
    dim3 block(N, N);

    a = (double *) malloc (size);
    b = (double *) malloc (size);
    c = (double *) malloc (size);
    
    cudaMalloc ((void**)&d_a, size);
    cudaMalloc ((void**)&d_b, size);
    cudaMalloc ((void**)&d_c, size);

    for( int i = 0; i < N*N; i++ )
    {
      a[i] = (double) ( rand() ) / ( RAND_MAX + 1.0 );
      b[i] = (double) ( rand() ) / ( RAND_MAX + 1.0 );
      c[i] = 0;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    for (int i=1; i<100; i++) {
        mm <<<grid, block>>> (d_a, d_b, d_c);
    }

    cudaDeviceSynchronize();
    cudaMemcpy (c, d_c, size, cudaMemcpyDeviceToHost);

    //for( int i=0; i < N*N; i++ )
    //{
    //    printf("%f\t%f\t%f\n", a[i], b[i], c[i]);
    //}
    printf("%f\n", c[N * N/2]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}

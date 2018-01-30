#include <stdio.h>
#include <cuda.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void mm(double *a, double *b, double *c)
{
    /* ----- YOUR CODE HERE ----- */

    /* -------------------------- */
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

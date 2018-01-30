#include <stdio.h>
#include <cuda.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void mm(double *A, double *B, double *C)
{
  
  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  const uint aBegin = N * BLOCK_SIZE * by;
  // Index of the last sub-matrix of A processed by the block
  const uint aEnd = aBegin + N - 1;
  // Step size used to iterate through the sub-matrices of A
  const uint aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  const uint bBegin = BLOCK_SIZE * bx;
  // Step size used to iterate through the sub-matrices of B
  const uint bStep = BLOCK_SIZE * N;

  // The element of the block sub-matrix that is computed
  // by the thread
  double Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) 
    {
      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      for (int k = 0; k < BLOCK_SIZE; ++k)
        Csub += A[a + N * ty + k] * B[b + N * k + tx];

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  const uint c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + N * ty + tx] = Csub;
}

int main () {
    
    double *a, *b, *c;
    double *d_a, *d_b, *d_c;
    double size = sizeof(double) * N*N;
    dim3 grid(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

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

    for (int i=0; i<100; i++) {
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

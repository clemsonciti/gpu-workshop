#include <stdio.h>
#include <malloc.h>

#define N 256
#define NSTEPS 50000
#define BLOCK_SIZE 32

__global__ void heat_conduction_step_gpu(double *d_T1, double *d_T2) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i > 0 && i < N-1 && j > 0 && j < N-1) {
        d_T1[i*N + j] = (
                    d_T2[(i-1)*N + j] +
                    d_T2[(i+1)*N + j] +
                    d_T2[i*N + (j-1)] +
                    d_T2[i*N + (j+1)]
                    ) / 4.0;
    }
    
    __syncthreads();

    if (i == N/2 && j == N/2) { 
        d_T1[i*N + j] = 1.0;
    }
}

int main () {
    
    double *T1, *T2;
    double *d_T1, *d_T2, *d_temp;
    int size = N *  N * sizeof(double);

    T1 = (double *) malloc (size);
    T2 = (double *) malloc (size);
    cudaMalloc ((void **) &d_T1, size);
    cudaMalloc ((void **) &d_T2, size);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) { 
            T1[i*N + j] = 0;
            T2[i*N + j] = 0;
        }
    }

    T1 [(N/2)*(N+1)] = 1.0;
    T2 [(N/2)*(N+1)] = 1.0;

    cudaMemcpy(d_T1, T1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T2, T2, size, cudaMemcpyHostToDevice);

    dim3 grid (N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 block (BLOCK_SIZE, BLOCK_SIZE);

    for (int t=0; t<NSTEPS; t++) {
        heat_conduction_step_gpu <<<grid, block>>> (d_T1, d_T2);
        d_temp = d_T1;
        d_T1 = d_T2;
        d_T2 = d_temp; 
    }

    cudaDeviceSynchronize();
    cudaMemcpy(T2, d_T2, size, cudaMemcpyDeviceToHost);
    printf("%0.10f\n", T2[N/4*N + N/4]);

    free(T1);
    free(T2);
    cudaFree(d_T1);
    cudaFree(d_T2);
    
    return 0;
}


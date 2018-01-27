#include <stdio.h>
#include <cuda.h>

__global__ void helloKernel() {
    printf("Hello from thread %d of block %d\n!", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello from the CPU\n");
    helloKernel <<<2, 4>>> ();
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    };
    return 0;
}

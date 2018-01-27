#include <stdio.h>
#include <cuda.h>

__global__ void helloKernel() {
    printf("Hello from thread %d of block %d\n!", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello from the CPU\n");
    helloKernel <<<1, 1024>>> ();
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    };
    return 0;
}

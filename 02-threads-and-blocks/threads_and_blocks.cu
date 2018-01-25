#include <stdio.h>
#include <cuda.h>

__global__ void helloKernel() {
    /* ----- YOUR CODE HERE ----- */

    /* -------------------------- */
}

int main() {
    printf("Hello from the CPU\n");
    /* ----- YOUR CODE HERE ----- */

    /* -------------------------- */
    cudaDeviceSynchronize();
    return 0;
}

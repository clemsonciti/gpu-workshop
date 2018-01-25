#include "cuda.h"
#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main()
{
    int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof( int );

	/* allocate space for host copies of a, b, c and setup input values */

	a = (int *)malloc( size );
	b = (int *)malloc( size );
	c = (int *)malloc( size );

	/* allocate space for device copies of a, b, c */
    /* ----- YOUR CODE HERE ----- */




    /* -------------------------- */

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	/* copy inputs to device */
    /* ----- YOUR CODE HERE ----- */




    /* -------------------------- */


	/* launch the kernel on the GPU */

	add<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

	/* copy result back to host */

	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

	printf( "c[0] = %d\n",0,c[0] );
	printf( "c[%d] = %d\n",N-1, c[N-1] );

	/* clean up */

	free(a);
	free(b);
	free(c);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} /* end main */


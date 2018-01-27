#include <stdio.h>
#include <cuda.h>

__global__ void sumKernel (double *d_a, double *d_b, double *d_c)
{
  /* Sums the values in arrays d_a and d_b,
     storing the result in d_c.
   */

  int i = threadIdx.x;
  d_c[i] = d_a[i] + d_b[i];
}

#define N 32

int main ()
{
  double *a, *b, *c;
  double *d_a, *d_b, *d_c;
  int size = N * sizeof (double);

  /* allocate space for host copies of a, b, c and setup input values */

  a = (double *) malloc (size);
  b = (double *) malloc (size);
  c = (double *) malloc (size);

  /* allocate space for device copies of a, b, c */
  /* ----- YOUR CODE HERE ----- */



  /* -------------------------- */

  for (int i = 0; i < N; i++)
  {
      a[i] = b[i] = i;
      c[i] = 0;
  }

  /* copy inputs to device */
  /* ----- YOUR CODE HERE ----- */



  /* -------------------------- */


  /* launch the kernel on the GPU */

  sumKernel <<< 1, N >>> (d_a, d_b, d_c);
  cudaDeviceSynchronize();
    
  /* copy result back to host */

  cudaMemcpy (c, d_c, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
      printf ("c[ %d ] = %f\n", i, c[i]);
  }

  /* clean up */

  free (a);
  free (b);
  free (c);
  cudaFree (d_a);
  cudaFree (d_b);
  cudaFree (d_c);

  return 0;
}

# Hello World on the GPU

In this exercise, you will write
the "Hello World" program on the GPU,
and run it on the cluster.

1.  Begin by requesting a GPU on Palmetto.

2.  Navigate to `~/gpu-workshop/01-gpu-hello`

3.  In the file called `hello_world.cu`, type out the following program:

    ```c
   	#include <stdio.h>
	#include <cuda.h>

	__global__ void helloKernel() {
		printf("Hello from the GPU!\n");
	}

	int main() {
    	printf("Hello from the CPU\n");
    	helloKernel <<<1, 1>>> ();
    	cudaDeviceSynchronize();
    return 0;
	} 
    ```

4. Load the `cuda-toolkit` module and compile the code `hello_world.cu`:

	```shellsession
	$ module load cuda-toolkit
	$ nvcc -o hello.out hello_world.cu
	```

5. Run the executable `hello.out`:

	```
	$ ./hello.out
	```

---


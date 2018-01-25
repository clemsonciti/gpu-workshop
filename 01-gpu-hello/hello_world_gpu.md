# Hello World on the GPU

In this exercise, you will write
the "Hello World" program on the GPU,
and run it on the cluster.

1.  Begin by requesting a GPU on Palmetto:

    ```shellsession
    $ qsub -I -l select=1:ncpus=2:mem=2gb:ngpus=1,walltime=1:00:00 -q R2387430
    ```

    Keep in mind the following:

    * The `ngpus=` option must be specified
    * Ask for *at least 2 cores*. For single core jobs, the rest of the PBS resource limits specification is ignored.
    You do not need more than 2 cores for this exercise.
    * The special queue `R2387430` is valid only for today. After today, you will not need to specify this option.

2.  Navigate to `~/gpu-workshop/01-gpu-hello`

3.  In a file called `hello_world.cu`, type out the following program:

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



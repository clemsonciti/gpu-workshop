# CUDA Threads and blocks

In this exercise, you will modify the CUDA kernel `helloKernel`
to print the thread ID and block ID for each
thread that executes the kernel.

1.  Navigate to `~/gpu-workshop/02-threads-and-blocks`.
2.  Make a copy of the program `hello_world.cu`,
    called `threads_and_blocks.cu`.

3.  Modify the kernel `helloKernel` to print the thread
    and block indices, by replacing the `printf` statement
    with the following:

    ```c
    printf("Hello from thread %d of block %d\n!", threadIdx.x, blockIdx.x);
    ```

4.  Set the number of blocks to 2 and the number of threads per
    block to 4 in the kernel call:

    ```c
    helloKernel <<<2, 4>> ();
    ```

5.  Compile the code and execute as before. Name the executable
    `threads_and_blocks.out`.

6.  Experiment with different values for the number of blocks
    and threads per block. What is the limit for the threads
    per block (hint: it's a multiple of 32!).

## Run the CUDA `deviceQuery` utility

How to find the limit on the number of threads per block?

The `deviceQuery` utility prints information about available GPUs on this node,
and can be run as follows:

```shellsession
/software/cuda-toolkit/8.0.44/samples/1_Utilities/deviceQuery/deviceQuery Starting...
```

Look for the line containing `Maximum number of threads per block`
to determine the maximum number of threads each block can contain.

## Checking for errors in CUDA

If we set the number of threads in the kernel call to 1025,
we didn't get an error.

When something goes wrong in a kernel,
no error is raised. The `cudaGetLastError` function can be
used to check for errors (if any). It's a good idea to
call this function after each kernel call:

```c
	
	...

    helloKernel <<<1, 1024>>> ();
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    };
    return 0;
```


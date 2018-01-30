# Heat Conduction on the GPU

In this exercise, you will implement the steady-state heat conduction
algorithm on the GPU.

There are two sections of the code in `heat_conduction.cu` you need to complete:

1. In the kernel `heat_conduction_step_gpu`,
write the code that assigns the value to `d_T1[i*N + j]`,
i.e., the new temperature at point `(i, j)`.
This value, is simply the average of the values of
the temperature at the points `(i-1, j)`, `(i+1, j)`, `(i, j-1)`,
and `(i, j+1)`. These values must be taken from the
array `d_T2`.

2. You also need to write the code for calling
the `heat_conduction_step_gpu` kernel.
The grid and block sizes are already provided
in variables `grid` and `block`.

3. `heat_conduction.c` provides a CPU implementation of the algorithm,
and can be compiled and run with:

```bash
$ gcc -o heat_cpu.out heat_conduction.c -std=c99
$ time ./heat_cpu.out
```

Also try adding the `-O2` switch:

```
$ gcc -O2 -o heat_cpu.out heat_conduction.c -std=c99
$ time ./heat_cpu.out
```

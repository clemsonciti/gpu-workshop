In this exercise,
you will write the kernel to compute the
matrix product on the GPU using
a **single block**.

Each thread will compute the value of `c(i,j)`
from the values of row `i` of matrix `a`
and column `j` of matrix `c`.

As a hint, your kernel must contain a single loop
over indices `k`:

```c
for (int k = 0; k < N; k++)
{
    c(i, j) += a(i, k) * b(k, j)
}
```



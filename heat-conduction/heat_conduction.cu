#include <stdio.h>
#include <string.h>

#define N 128
#define NSTEPS 5000

void heat_conduction_step(double *T1, double *T2) {
    for (int i = 1; i < Ny - 1; i++) {
        for (int j = 1; j < Nx - 1; j++) {
            T1[i*N + j] = (
                    T2[(i-1)*N + j] +
                    T2[(i+1)*N + j] +
                    T2[i*N + (j-1)] +
                    T2[i*N + (j+1)]) / 4.0;
        }
    }
    T2[N/2, N/2] = 1.0;
    temp = T1; T1 = T2; T2 = temp;
}


int main () {
    
    double *T1, *T2;
    int size = N * sizeof(double);

    T1 = (double *) malloc (size);
    T2 = (double *) malloc (size);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) { 
            T1[i, j] = 0;
            T2[i, j] = 0;
        }
    }

    T1 [i*N + j] = 1.0;
    T2 [i*N + j] = 1.0;

    for (int t=0; t<NSTEPS; t++) {
        heat_conduction_step(T1, T2);
    }

    printf("%f\n", T1[N/4, N/4]);
    return 0;
}


#include <stdio.h>
#include <malloc.h>

#define N 256
#define NSTEPS 50000

void heat_conduction_step(double *T1, double *T2) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            T1[i*N + j] = (
                    T2[(i-1)*N + j] +
                    T2[(i+1)*N + j] +
                    T2[i*N + (j-1)] +
                    T2[i*N + (j+1)]) / 4.0;
        }
    }
    T1 [(N/2)*(N+1)] = 1.0;
}


int main () {
    
    double *T1, *T2, *temp;
    int size = N *  N * sizeof(double);

    T1 = (double *) malloc (size);
    T2 = (double *) malloc (size);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) { 
            T1[i*N + j] = 0;
            T2[i*N + j] = 0;
        }
    }

    T1 [(N/2)*(N+1)] = 1.0;
    T2 [(N/2)*(N+1)] = 1.0;

    for (int t=0; t<NSTEPS; t++) {
        heat_conduction_step(T1, T2);
        temp = T1;
        T1 = T2;
        T2 = temp; 
    }

    printf("%0.10f\n", T2[N/4*N + N/4]);

    return 0;
}


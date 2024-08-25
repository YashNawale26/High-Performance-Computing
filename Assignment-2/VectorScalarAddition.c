#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Adjust this for different data sizes

void vector_scalar_addition(double *vector, double scalar, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vector[i] += scalar;
    }
}

int main() {
    double *vector = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double scalar = 3.14;

    // Initialize vector
    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector[i] = (double)i;
    }

    double start_time, end_time;

    // Test with different thread counts
    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        omp_set_num_threads(num_threads);
        
        start_time = omp_get_wtime();
        vector_scalar_addition(vector, scalar, VECTOR_SIZE);
        end_time = omp_get_wtime();

        printf("Threads: %d, Time: %f seconds\n", num_threads, end_time - start_time);
    }

    free(vector);
    return 0;
}
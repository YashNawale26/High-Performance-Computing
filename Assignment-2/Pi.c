#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_STEPS 1000000000 // Adjust this for different data sizes

double calculate_pi(long long num_steps) {
    double pi = 0.0;
    double step = 1.0 / (double)num_steps;

    #pragma omp parallel
    {
        double x, sum = 0.0;
        #pragma omp for
        for (long long i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
        #pragma omp atomic
        pi += sum;
    }

    pi *= step;
    return pi;
}

int main() {
    double start_time, end_time, pi;

    // Test with different thread counts
    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        omp_set_num_threads(num_threads);
        
        start_time = omp_get_wtime();
        pi = calculate_pi(NUM_STEPS);
        end_time = omp_get_wtime();

        printf("Threads: %d, Pi: %.15f, Time: %f seconds\n", num_threads, pi, end_time - start_time);
    }

    return 0;
}
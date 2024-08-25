#include <stdio.h>
#include <omp.h>

#define THRESHOLD 30
#define DISPLAY_LIMIT 10

long long fibonacci_sequential(int n) {
    if (n <= 1) return n;
    return fibonacci_sequential(n - 1) + fibonacci_sequential(n - 2);
}

long long fibonacci_parallel(int n) {
    if (n <= THRESHOLD) {
        return fibonacci_sequential(n);
    }

    long long x, y;

    #pragma omp task shared(x)
    x = fibonacci_parallel(n - 1);

    #pragma omp task shared(y)
    y = fibonacci_parallel(n - 2);

    #pragma omp taskwait

    return x + y;
}

void print_fibonacci_sequence(int n, long long result) {
    printf("Fibonacci Sequence (first %d numbers):\n", DISPLAY_LIMIT);
    long long a = 0, b = 1;
    for (int i = 0; i < DISPLAY_LIMIT && i <= n; i++) {
        if (i == n) {
            printf("%lld", result);
        } else if (i < 2) {
            printf("%lld", i);
        } else {
            long long temp = a + b;
            a = b;
            b = temp;
            printf("%lld", temp);
        }
        if (i < DISPLAY_LIMIT - 1 && i < n) printf(", ");
    }
    if (n >= DISPLAY_LIMIT) printf(", ...");
    printf("\n");
}

int main() {
    int n = 45;
    double start_time, end_time;

    printf("Computing Fibonacci(%d)\n\n", n);

    // Sequential computation
    start_time = omp_get_wtime();
    long long result_seq = fibonacci_sequential(n);
    end_time = omp_get_wtime();
    printf("Sequential Result:\n");
    print_fibonacci_sequence(n, result_seq);
    printf("F(%d) = %lld\n", n, result_seq);
    printf("Sequential Time: %.6f seconds\n\n", end_time - start_time);

    // Parallel computation
    start_time = omp_get_wtime();
    long long result_par;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result_par = fibonacci_parallel(n);
        }
    }
    end_time = omp_get_wtime();
    printf("Parallel Result:\n");
    print_fibonacci_sequence(n, result_par);
    printf("F(%d) = %lld\n", n, result_par);
    printf("Parallel Time: %.6f seconds\n", end_time - start_time);

    return 0;
}
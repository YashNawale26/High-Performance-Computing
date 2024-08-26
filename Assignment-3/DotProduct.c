#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 100000

void print_array(int arr[], int n, const char* name) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int a[N], b[N];  // Using int for the arrays since the elements are small integers
    long long int sequential_product = 0;  // long long int to store large sums
    long long int parallel_product = 0;    // long long int to store large sums

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;  // Random values between 0 and 99
        b[i] = rand() % 100;  // Random values between 0 and 99
    }

    // Optional: Uncomment these lines if you want to print the arrays
    // printf("Array A: ");
    // print_array(a, N, "Array a");
    // printf("\n");
    // printf("Array B: ");
    // print_array(b, N, "Array b");

/********************************************************/

    clock_t sequential_start_time = clock();  // Using clock_t to measure time
    
    printf("Sequential Execution\n");
    for (int i = 0; i < N; i++) {
        sequential_product += (long long int)a[i] * b[i];  // long long int for large multiplication
    }
    clock_t sequential_end_time = clock();

    double sequential_time = (double)(sequential_end_time - sequential_start_time) / CLOCKS_PER_SEC;

    printf("Scalar product (sequential): %lld\n", sequential_product);
    printf("Time taken (sequential): %f seconds\n", sequential_time);

/********************************************************/

/********************************************************/

    double parallel_start_time = omp_get_wtime();  // Using omp_get_wtime() for high-precision timing
    
    printf("Parallel Execution\n"); 

    #pragma omp parallel for reduction(+:parallel_product) schedule(static)
    for (int i = 0; i < N; i++) {
        parallel_product += (long long int)a[i] * b[i];  // long long int for large multiplication
    }

    double parallel_end_time = omp_get_wtime(); 
    double parallel_time = parallel_end_time - parallel_start_time;  // double for precise time calculation
    
    printf("Scalar product (parallel): %lld\n", parallel_product);
    printf("Time taken (parallel): %f seconds\n", parallel_time);

/********************************************************/

    double speedup = sequential_time / parallel_time;  // double for accurate speedup calculation
    printf("Speed Up: %f\n", speedup);

    return 0;
}

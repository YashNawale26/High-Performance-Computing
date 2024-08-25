#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

#define N 100000  // Size of vectors

void sort_ascending(int arr[], int n) {
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void sort_descending(int arr[], int n) {
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] < arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j] = temp;
            }
        }
    }
}

int main() {
    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    long long min_product = 0;

    // Initialize arrays with random values
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    double start_time = omp_get_wtime();

    // Sort array 'a' in ascending order and 'b' in descending order
    #pragma omp sections nowait
    {
        #pragma omp section
        sort_ascending(a, N);

        #pragma omp section
        sort_descending(b, N);
    }

    // Calculate the minimum scalar product
    #pragma omp parallel for reduction(+:min_product) schedule(guided) ordered
    for (int i = 0; i < N; i++) {
        #pragma omp ordered
        min_product += (long long)a[i] * b[i];
    }

    double end_time = omp_get_wtime();

    printf("Minimum scalar product: %lld\n", min_product);
    printf("Time taken: %f seconds\n", end_time - start_time);

    free(a);
    free(b);
    return 0;
}
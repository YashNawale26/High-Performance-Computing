#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10  // Size of vectors for simplicity

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
                arr[j + 1] = temp;
            }
        }
    }
}

void print_array(int arr[], int n, const char* name) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int a[N], b[N];
    long long min_product = 0;

    // Initialize arrays with simple values for easy calculation
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;  // Simple values from 1 to N
        b[i] = N - i;  // Simple values from N to 1
    }

    // Print the initial arrays
    print_array(a, N, "Array a");
    print_array(b, N, "Array b");

    double start_time = omp_get_wtime();

    // Sort array 'a' in ascending order and 'b' in descending order
    #pragma omp sections nowait
    {
        #pragma omp section
        sort_ascending(a, N);

        #pragma omp section
        sort_descending(b, N);
    }

    // Print the sorted arrays
    print_array(a, N, "Sorted array a");
    print_array(b, N, "Sorted array b");

    // Calculate the minimum scalar product
    #pragma omp parallel for reduction(+:min_product) schedule(guided)
    for (int i = 0; i < N; i++) {
        min_product += (long long)a[i] * b[i];
    }

    double end_time = omp_get_wtime();

    printf("Minimum scalar product: %lld\n", min_product);
    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}

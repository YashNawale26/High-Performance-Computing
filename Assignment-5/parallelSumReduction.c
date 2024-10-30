#include <stdio.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

int main() {
    int array[ARRAY_SIZE];
    long long total_sum = 0;

    // Initialize array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i + 1; // Sample data
    }

    #pragma omp parallel for reduction(+:total_sum)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        total_sum += array[i];
    }

    printf("Total sum: %lld\n", total_sum);
    return 0;
}

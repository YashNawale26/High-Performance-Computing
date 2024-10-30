#include <stdio.h>
#include <omp.h>

int main() {
    int sum = 0;

    #pragma omp parallel reduction(+:sum)
    {
        int thread_id = omp_get_thread_num();
        int square = thread_id * thread_id;
        printf("Thread %d: Square = %d\n", thread_id, square);
        sum += square;
    }

    printf("Sum of squares of thread IDs: %d\n", sum);
    return 0;

}   
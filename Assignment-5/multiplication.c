#include <stdio.h>
#include <omp.h>

int main() {
    int Aryabhatta = 10;

    #pragma omp parallel private(Aryabhatta)
    {
        int thread_id = omp_get_thread_num();
        int result = thread_id * Aryabhatta;
        printf("Thread %d: %d * %d = %d\n", thread_id, thread_id, Aryabhatta, result);
    }

    return 0;   
}

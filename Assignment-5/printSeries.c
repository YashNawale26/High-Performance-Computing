#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("2\n");
        }
        #pragma omp section
        {
            printf("4\n");
        }
    }
    return 0;
}

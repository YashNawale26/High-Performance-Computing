#include <stdio.h>
#include <omp.h>

int main() {
    const char *family_members[] = {"Samyak", "Paras", "Alka", "Sunil"};
    int num_members = sizeof(family_members) / sizeof(family_members[0]);

    #pragma omp parallel for
    for (int i = 0; i < num_members; i++) {
        int thread_id = omp_get_thread_num();
        printf("Thread %d: %s\n", thread_id, family_members[i]);
    }

    return 0;   
}

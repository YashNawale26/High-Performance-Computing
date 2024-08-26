#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void add_matrices(int **A, int **B, int **C, int size, int num_threads)
{
#pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

double measure_time(int **A, int **B, int **C, int size, int num_threads)
{
    double start_time = omp_get_wtime();
    add_matrices(A, B, C, size, num_threads);
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main()
{
    int sizes[] = {250, 500, 750, 1000, 2000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    srand(time(NULL));

    int thread_counts[] = {1, 2, 4, 8};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int s = 0; s < num_sizes; s++)
    {
        int size = sizes[s];
        printf("Matrix size: %d x %d\n", size, size);

        // Allocate matrices
        // int **A = (int **)malloc(size * sizeof(int *));
        // int **B = (int **)malloc(size * sizeof(int *));
        // int **C = (int **)malloc(size * sizeof(int *));
        // for (int i = 0; i < size; i++)
        // {
        //     A[i] = (int *)malloc(size * sizeof(int));
        //     B[i] = (int *)malloc(size * sizeof(int));
        //     C[i] = (int *)malloc(size * sizeof(int));
        // }

        int A[sizes[s]][sizes[s]];
        int B[sizes[s]][sizes[s]];
        int C[sizes[s]][sizes[s]];

        // initialize_matrix A
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                A[i][j] = rand() % 100;
            }
        }

        // initialize_matrix B
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                B[i][j] = rand() % 100;
            }
        }

        double base_time = 0;

        for (int t = 0; t < num_thread_counts; t++)
        {
            int num_threads = thread_counts[t];
            double execution_time = measure_time(A, B, C, size, num_threads);

            if (num_threads == 1)
            {
                base_time = execution_time;
            }

            double speedup = base_time / execution_time;

            printf("Threads: %d, Time: %f seconds, Speedup: %f\n",
                   num_threads, execution_time, speedup);
        }
        
        // Free allocated memory
        for (int i = 0; i < size; i++)
        {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }
        free(A);
        free(B);
        free(C);

        printf("\n");
    }

    return 0;
}
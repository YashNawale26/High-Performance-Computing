#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    double start_time, end_time;
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n;
    // Process 0 gets the matrix size from user
    if (world_rank == 0) {
        printf("Enter the size of matrices (n x n): ");
        scanf("%d", &n);
    }
    // Broadcast the size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Dynamically allocate matrices
    int (*A)[n] = malloc(sizeof(int[n][n]));
    int (*B)[n] = malloc(sizeof(int[n][n]));
    int (*C)[n] = malloc(sizeof(int[n][n]));

    // Process 0 gets input for matrices A and B
    if (world_rank == 0) {
        printf("Enter matrix A (%dx%d):\n", n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%d", &A[i][j]);
            }
        }

        printf("Enter matrix B (%dx%d):\n", n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                scanf("%d", &B[i][j]);
            }
        }

        printf("\nMatrix A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", A[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", B[i][j]);
            }
            printf("\n");
        }
    }

    int local_rows = n / world_size;
    int (*local_A)[n] = malloc(sizeof(int[local_rows][n]));
    int (*local_B)[n] = malloc(sizeof(int[n][n]));
    int (*local_C)[n] = malloc(sizeof(int[local_rows][n]));

    // Start timing
    start_time = MPI_Wtime();

    // Scatter matrix A and broadcast matrix B
    MPI_Scatter(A, local_rows * n, MPI_INT, local_A, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local matrix multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Gather the results
    MPI_Gather(local_C, local_rows * n, MPI_INT, C, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    // End timing
    end_time = MPI_Wtime();

    // Process 0 prints the result matrix C and execution time
    if (world_rank == 0) {
        printf("\nResulting matrix C:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
        printf("\nExecution time: %f seconds\n", end_time - start_time);
    }

    // Free allocated memory
    free(local_A);
    free(local_B);
    free(local_C);
    if (world_rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
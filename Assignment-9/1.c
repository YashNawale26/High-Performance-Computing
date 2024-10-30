#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 4;
    int A[4][4] = { 
        {1, 2, 3, 4}, 
        {5, 6, 7, 8}, 
        {9, 10, 11, 12}, 
        {13, 14, 15, 16} 
    };
    int x[4] = {1, 1, 1, 1};  
    int local_rows = n / world_size;  
    int local_result[local_rows];

    // MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
    int local_A[local_rows][n];
    MPI_Scatter(A, local_rows * n, MPI_INT, local_A, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    // MPI_Bcast(buffer, count, datatype, root, comm)
    MPI_Bcast(x, n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0;
        for (int j = 0; j < n; j++) {
            local_result[i] += local_A[i][j] * x[j];
        }
    }

    int final_result[n];
    // MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm)
    MPI_Gather(local_result, local_rows, MPI_INT, final_result, local_rows, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Resulting vector: \n");
        for (int i = 0; i < n; i++) {
            printf("%d\n", final_result[i]);
        }
    }

    MPI_Finalize();
    return 0;
}

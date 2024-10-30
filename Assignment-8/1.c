#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MESSAGE_SIZE 10000000  

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  

    int* message = (int*)malloc(MESSAGE_SIZE * sizeof(int));  

    if (world_rank == 0) {
        MPI_Send(message, MESSAGE_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message to Process 1\n");

        MPI_Recv(message, MESSAGE_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received message from Process 1\n");
    } else if (world_rank == 1) {
        MPI_Send(message, MESSAGE_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Process 1 sent message to Process 0\n");

        MPI_Recv(message, MESSAGE_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message from Process 0\n");
    }

    free(message); 
    MPI_Finalize();  
    return 0;
}


// int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
// int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

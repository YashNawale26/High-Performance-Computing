// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// void matrixMultiplySerial(int **A, int **B, int **C, int N) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             C[i][j] = 0;
//             for (int k = 0; k < N; k++) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

// int main() {
//     int N = 500; // Example matrix size
//     int **A = (int **)malloc(N * sizeof(int *));
//     int **B = (int **)malloc(N * sizeof(int *));
//     int **C = (int **)malloc(N * sizeof(int *));
//     for (int i = 0; i < N; i++) {
//         A[i] = (int *)malloc(N * sizeof(int));
//         B[i] = (int *)malloc(N * sizeof(int));
//         C[i] = (int *)malloc(N * sizeof(int));
//     }

//     // Initialize matrices with some values
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             A[i][j] = rand() % 100;
//             B[i][j] = rand() % 100;
//         }
//     }

//     clock_t start = clock();
//     matrixMultiplySerial(A, B, C, N);
//     clock_t end = clock();

//     printf("Serial Execution Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

//     // Free memory
//     for (int i = 0; i < N; i++) {
//         free(A[i]);
//         free(B[i]);
//         free(C[i]);
//     }
//     free(A);
//     free(B);
//     free(C);

//     return 0;
// }










// MATRIX MULTIPLICATION USING STATIC SCHEDULING


// #include <stdio.h>
// #include <stdlib.h>
// #include <omp.h>

// void matrixMultiplyStatic(int **A, int **B, int **C, int N) {
//     #pragma omp parallel for schedule(static)
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             C[i][j] = 0;
//             for (int k = 0; k < N; k++) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

// int main() {
//     int N = 500; // Example matrix size
//     int **A = (int **)malloc(N * sizeof(int *));
//     int **B = (int **)malloc(N * sizeof(int *));
//     int **C = (int **)malloc(N * sizeof(int *));
//     for (int i = 0; i < N; i++) {
//         A[i] = (int *)malloc(N * sizeof(int));
//         B[i] = (int *)malloc(N * sizeof(int));
//         C[i] = (int *)malloc(N * sizeof(int));
//     }

//     // Initialize matrices with some values
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             A[i][j] = rand() % 100;
//             B[i][j] = rand() % 100;
//         }
//     }

//     double start = omp_get_wtime();
//     matrixMultiplyStatic(A, B, C, N);
//     double end = omp_get_wtime();

//     printf("Static Scheduling Execution Time: %f seconds\n", end - start);

//     // Free memory
//     for (int i = 0; i < N; i++) {
//         free(A[i]);
//         free(B[i]);
//         free(C[i]);
//     }
//     free(A);
//     free(B);
//     free(C);

//     return 0;
// }





// MATRIX MULTIPLICATION USING DYNAMIC SCHEDULING

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrixMultiplyDynamic(int **A, int **B, int **C, int N) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int N = 500; // Example matrix size
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));
    }

    // Initialize matrices with some values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    double start = omp_get_wtime();
    matrixMultiplyDynamic(A, B, C, N);
    double end = omp_get_wtime();

    printf("Dynamic Scheduling Execution Time: %f seconds\n", end - start);

    // Free memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

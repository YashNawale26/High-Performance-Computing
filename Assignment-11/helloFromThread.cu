#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromThreads() {
    int threadId = threadIdx.x;  // Get the thread ID within the block
    printf("Hello World from Thread ID: %d\n", threadId);
}

int main() {
    int numThreads = 10;  // Number of threads per block

    // Launch the kernel with 1 block and numThreads threads
    helloFromThreads<<<1, numThreads>>>();

    // Synchronize to wait for all threads to finish
    cudaDeviceSynchronize();

    return 0;
}

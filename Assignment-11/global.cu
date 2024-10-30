#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromThreads() {
    // Calculate global thread ID (threadIdx.x + blockIdx.x * blockDim.x)
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from Global Thread ID: %d\n", globalThreadId);
}

int main() {
    int numBlocks = 2;      // Number of blocks
    int numThreads = 5;     // Number of threads per block

    // Launch the kernel with numBlocks and numThreads per block
    helloFromThreads<<<numBlocks, numThreads>>>();

    // Synchronize to wait for all threads to finish
    cudaDeviceSynchronize();

    return 0;
}

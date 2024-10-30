#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFrom2DThreads() {
    // Get the 2D block index and thread index
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    // Print the block and thread IDs in 2D
    printf("Hello World from Block(%d, %d) Thread(%d, %d)\n", blockX, blockY, threadX, threadY);
}

int main() {
    // Define 2D block and thread dimensions
    dim3 numBlocks(2, 2);      // 2x2 blocks
    dim3 numThreads(3, 3);     // 3x3 threads per block

    // Launch the kernel with 2D blocks and 2D threads
    helloFrom2DThreads<<<numBlocks, numThreads>>>();

    // Synchronize to wait for all threads to finish
    cudaDeviceSynchronize();

    return 0;
}

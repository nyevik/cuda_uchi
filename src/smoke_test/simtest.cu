#include <cuda_runtime.h>
#include <iostream>

__global__ void test1()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d, thread %d, block %d * %d\n", tid, threadIdx.x, blockIdx.x, blockDim.x);
}

int main()
{
    int numBlocks = 2;
    int threadsPerBlock = 5;
    test1<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}
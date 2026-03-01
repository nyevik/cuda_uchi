#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp size: " << deviceProp.warpSize << " threads" << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max grid size: [" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]" << std::endl;
    std::cout << "Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Multi-processor count: " << deviceProp.multiProcessorCount << std::endl;
    return 0;       

}
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaDeviceProp p{};
    err = cudaGetDeviceProperties(&p, dev);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Device: " << p.name << "\n";
    std::cout << "Compute capability: " << p.major << "." << p.minor << "\n";
    std::cout << "SM count (multiProcessorCount): " << p.multiProcessorCount << "\n";
    std::cout << "Warp size: " << p.warpSize << "\n";
    std::cout << "Max threads per SM: " << p.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max threads per block: " << p.maxThreadsPerBlock << "\n";
    return 0;
}
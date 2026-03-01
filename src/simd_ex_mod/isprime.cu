// isprime.cu
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err__ = (call);                                     \
        if (err__ != cudaSuccess)                                       \
        {                                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)    \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// Same algorithm for CPU and GPU: 6k±1, overflow-safe bound (i <= n/i).
__host__ __device__ inline bool isPrime_6k1(long long n)
{
    if (n <= 1)
        return false;
    if (n <= 3)
        return true; // 2,3
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    // test factors 5,7,11,13,17,19,... (i and i+2, step 6)
    for (long long i = 5; i <= n / i; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

__global__ void isPrimeKernel(const long long *numbers, std::uint8_t *results, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    long long n = numbers[idx];
    results[idx] = static_cast<std::uint8_t>(isPrime_6k1(n));
}

int main()
{
    // ---- Input range (make the test set explicit) ----
    const long long start = 100'001LL; // odd
    const long long end = 190'001LL;   // odd

    // build the exact same list for CPU and GPU
    const int count = static_cast<int>((end - start) / 2 + 1); // odds only
    std::vector<long long> h_numbers(count);
    for (int i = 0; i < count; ++i)
    {
        h_numbers[i] = start + static_cast<long long>(i) * 2;
    }

    // ---- CPU baseline (compute only) ----
    std::vector<std::uint8_t> h_cpu(count);
    auto cpu_t0 = std::chrono::high_resolution_clock::now();

    unsigned long long cpu_prime_count = 0;
    for (int i = 0; i < count; ++i)
    {
        const bool p = isPrime_6k1(h_numbers[i]);
        h_cpu[i] = static_cast<std::uint8_t>(p);
        cpu_prime_count += h_cpu[i];
    }

    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_t1 - cpu_t0;

    // ---- GPU path (compute comparable + end-to-end) ----
    long long *d_numbers = nullptr;
    std::uint8_t *d_results = nullptr;

    CUDA_CHECK(cudaMalloc(&d_numbers, sizeof(long long) * count));
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(std::uint8_t) * count));

    // Events for timing
    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    // Warm-up to avoid first-launch/context noise
    CUDA_CHECK(cudaMemcpy(d_numbers, h_numbers.data(), sizeof(long long) * count, cudaMemcpyHostToDevice));
    isPrimeKernel<<<blocks, threads>>>(d_numbers, d_results, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time end-to-end: H2D + kernel + D2H
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_numbers, h_numbers.data(), sizeof(long long) * count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    // Time kernel-only (compute comparable to CPU loop)
    CUDA_CHECK(cudaEventRecord(e2));
    isPrimeKernel<<<blocks, threads>>>(d_numbers, d_results, count);
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));

    std::vector<std::uint8_t> h_gpu(count);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_results, sizeof(std::uint8_t) * count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h2d_ms = 0.0f, kernel_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e2, e3));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e3)); // H2D + kernel (D2H not included here)
    // If you want strictly "full end-to-end incl D2H", time around the D2H too; see note below.

    // Count primes on GPU results (forces use of output)
    unsigned long long gpu_prime_count = 0;
    for (int i = 0; i < count; ++i)
        gpu_prime_count += h_gpu[i];

    // Optional correctness check
    int mismatches = 0;
    for (int i = 0; i < count; ++i)
    {
        if (h_cpu[i] != h_gpu[i])
        {
            if (++mismatches <= 5)
            {
                std::cerr << "Mismatch at i=" << i << " n=" << h_numbers[i]
                          << " cpu=" << int(h_cpu[i]) << " gpu=" << int(h_gpu[i]) << "\n";
            }
        }
    }

    // ---- Report ----
    std::cout << "Range: [" << start << ", " << end << "] odds only, count=" << count << "\n";
    std::cout << "CPU prime count: " << cpu_prime_count << "\n";
    std::cout << "GPU prime count: " << gpu_prime_count << "\n";
    std::cout << "Mismatches: " << mismatches << "\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU compute time:        " << cpu_ms.count() << " ms\n";
    std::cout << "GPU H2D time:            " << h2d_ms << " ms\n";
    std::cout << "GPU kernel time:         " << kernel_ms << " ms   (comparable to CPU compute)\n";
    std::cout << "GPU H2D+kernel time:     " << total_ms << " ms\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(e3));
    CUDA_CHECK(cudaFree(d_numbers));
    CUDA_CHECK(cudaFree(d_results));

    return (mismatches == 0) ? 0 : 2;
}
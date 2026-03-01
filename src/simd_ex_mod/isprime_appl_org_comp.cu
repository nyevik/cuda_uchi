
/**
 * @file isprime.cu
 * @author Nikolai Yevik
 * @brief This file contains the implementation of the isPrime function for both CPU and GPU.
 *        The GPU implementation uses CUDA to check for prime numbers in parallel, while the
 *        CPU implementation uses a simple loop to check for primality.
 * @version 1.0
 */

#include <iostream>
#include <chrono>

/**
 * @brief Checks if a number is prime on the CPU.
 * @param num The number to check for primality.
 * @return true if the number is prime, false otherwise.
 * @note Prima number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers,
 * or, in other words, the number that has no positive divisors other than 1 and itself.
 */
bool isPrimeCpu(long long num)
{ //////// This function is not optimized for large numbers, but it serves as a baseline for comparison with the GPU implementation.

    if (num <= 1)
        return false;
    if (num <= 3)
        return true;
    if (num % 2 == 0 || num % 3 == 0)
        return false;

    for (long long i = 5; i <= num / i; i += 6)
    {
        if (num % i == 0 || num % (i + 2) == 0)
            return false;
    }
    return true;
}

/**
 * @brief Checks if numbers are prime on the GPU.
 * @param numbers Array of numbers to check for primality.
 * @param results Array to store the results (true if prime, false otherwise).
 * @param count Number of elements in the arrays.
 */
__global__ void isPrimeGpu(long long *numbers, bool *results, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        long long num = numbers[idx];
        if (num <= 1)
            results[idx] = false;
        else if (num <= 3)
            results[idx] = true;
        else if (num % 2 == 0 || num % 3 == 0)
            results[idx] = false;
        else
        {
            bool isPrime = true;
            for (long long i = 5; i <= num / i; i += 6)
            {
                if (num % i == 0 || num % (i + 2) == 0)
                {
                    isPrime = false;
                    break;
                }
            }
            results[idx] = isPrime;
        }
    }
}

/*__global__ void isPrimeGpu(long long start, long long end)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long num = start + idx * 2; // Check only odd numbers

    if (num <= end)
    {
        if (num <= 1)
            printf("%lld is not prime\n", num);
        else if (num <= 3)
            printf("%lld is prime\n", num);
        else if (num % 2 == 0 || num % 3 == 0)
            printf("%lld is not prime\n", num);
        else
        {
            bool isPrime = true;
            for (long long i = 5; i <= num / i; i += 6)
            {
                if (num % i == 0 || num % (i + 2) == 0)
                {
                    isPrime = false;
                    break;
                }
            }
            if (isPrime)
                printf("%lld is prime\n", num);
            else
                printf("%lld is not prime\n", num);
        }
    }
}*/

/**
 * @brief Checks if numbers are prime on the GPU without using arrays, directly calculating the number based on thread index.
 * @param start The starting number to check for primality.
 * @param end The ending number to check for primality.
 * @note This kernel assumes that the range of numbers is large enough to utilize
 * the GPU threads effectively and that the starting number is odd to avoid checking even numbers (except for 2).
 * ***This is a test implementation, not production ready!***
 */
__global__ void isPrimeGpu(long long start, long long end)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long num = start + tid * 2; // Check only odd numbers
    if (num > end)
        return;          // Out of range check
    bool isPrime = true; // Assume prime until proven otherwise
    if (num <= 1)
    { // 0 and 1 are not prime
        isPrime = false;
        return;
    }
    else if (num <= 3)
    { // 2 and 3 are prime
        isPrime = true;
        return;
    }
    else if (num % 2 == 0 || num % 3 == 0)
    { // Eliminate multiples of 2 and 3
        isPrime = false;
        return;
    }
    for (long long i = 3; i * i <= num; i += 2)
    { // overflow check for i*i needed for large num!!!
        if (num % i == 0)
        {
            isPrime = false; // branching occurs here when a divisor is found, but this is necessary to determine primality
            // should not have real impact on performance as it just sets isPrime and returns.
            break;
        }
    }
}
int main()
{
    long long start = 100'001LL; // must start with odd
    long long end = 190'001LL;

    int threadsPerBlock = 1024;                                                 // maximum for GEFORCE RTX 5060
    int totalNumbers = (end - start) / 2 + 1;                                   // only odd numbers are checked
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock; // calculate the number of blocks needed to cover all numbers

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    isPrimeGpu<<<blocksPerGrid, threadsPerBlock>>>(start, end);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2)
    {
        isPrimeCpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
} // end of main
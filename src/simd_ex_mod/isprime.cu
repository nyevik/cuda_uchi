
/**
 * @file isprime.cu
 * @author Nikolai Yevik
 * @brief This file contains the implementation of the isPrime function for both CPU and GPU.
 *        The GPU implementation uses CUDA to check for prime numbers in parallel, while the
 *        CPU implementation uses a simple loop to check for primality.
 * @version 1.0
 */

/**
 * @brief Checks if a number is prime on the CPU.
 * @param num The number to check for primality.
 * @return true if the number is prime, false otherwise.
 * @note Prima number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers,
 * or, in other words, the number that has no positive divisors other than 1 and itself.
 */
#include <iostream>
#include <chrono>

bool isPrimeCpu(long long num)
{

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

int main()
{
    long long start = 100'001LL; // must start with odd
    long long end = 190'001LL;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2)
    {
        isPrimeCpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
} // end of main
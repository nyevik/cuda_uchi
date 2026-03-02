
/********************************************************************
 *  Vector addition – modern C++ version
 *
 *  • Host memory is managed by std::vector (RAII, bounds‑checked).
 *  • Device memory is wrapped in a small RAII class (DeviceMemory).
 *  • CUDA errors are turned into C++ exceptions via the CUDA_CHECK macro.
 *  • No manual free() or cudaFree() – everything cleans up automatically.
 ********************************************************************/

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------
 *  Simple CUDA error‑checking macro.
 *  Throws std::runtime_error on failure, including file/line info.
 * ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            throw std::runtime_error(std::string("CUDA error at ") +                     \
                                     __FILE__ + ":" + std::to_string(__LINE__) + " – " + \
                                     cudaGetErrorString(err));                           \
        }                                                                                \
    } while (0)

/* ------------------------------------------------------------------
 *  RAII wrapper for device memory.
 *  Non‑copyable, movable.
 * ------------------------------------------------------------------ */
template <typename T>
class DeviceMemory
{
public:
    explicit DeviceMemory(std::size_t count) : count_(count), ptr_(nullptr)
    {
        if (count_ == 0)
            return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_),
                              count_ * sizeof(T)));
    }

    // Delete copy semantics – we own the allocation.
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;

    // Move semantics – transfer ownership.
    DeviceMemory(DeviceMemory &&other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceMemory &operator=(DeviceMemory &&other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceMemory() { release(); }

    // Accessors
    T *get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

private:
    void release() noexcept
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    T *ptr_;
    std::size_t count_;
};

/* ------------------------------------------------------------------
 *  CUDA kernel – unchanged from the original example.
 * ------------------------------------------------------------------ */
__global__ void vectorAddKernel(const float *A,
                                const float *B,
                                float *C,
                                int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/* ------------------------------------------------------------------
 *  CPU reference implementation (used for verification).
 * ------------------------------------------------------------------ */
void vectorAddCpu(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C)
{
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   [](float a, float b)
                   { return a + b; });
}

/* ------------------------------------------------------------------
 *  Main – demonstrates allocation, copy, kernel launch, and cleanup.
 * ------------------------------------------------------------------ */
int main()
{
    try
    {
        const std::size_t N = 100'000'000; // 100 M elements
        const std::size_t bytes = N * sizeof(float);

        // ---------- Host buffers (RAII via std::vector) ----------
        std::vector<float> h_A(N), h_B(N), h_C(N);
        // Simple initialization – feel free to replace with random data.
        std::fill(h_A.begin(), h_A.end(), 1.0f);
        std::fill(h_B.begin(), h_B.end(), 2.0f);

        // ---------- Device buffers (RAII wrapper) ----------
        DeviceMemory<float> d_A(N);
        DeviceMemory<float> d_B(N);
        DeviceMemory<float> d_C(N);

        // ---------- Transfer host → device ----------
        CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), bytes,
                              cudaMemcpyHostToDevice));

        // ---------- Kernel launch ----------
        const int threadsPerBlock = 256;
        const int blocks = static_cast<int>((N + threadsPerBlock - 1) /
                                            threadsPerBlock);
        vectorAddKernel<<<blocks, threadsPerBlock>>>(d_A.get(),
                                                     d_B.get(),
                                                     d_C.get(),
                                                     static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------- Transfer device → host ----------
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.get(), bytes,
                              cudaMemcpyDeviceToHost));

        // ---------- Verify result ----------
        bool correct = true;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (h_C[i] != h_A[i] + h_B[i])
            {
                std::cerr << "Mismatch at index " << i
                          << ": " << h_C[i] << " != "
                          << h_A[i] + h_B[i] << '\n';
                correct = false;
                break;
            }
        }
        std::cout << (correct ? "Result correct!\n"
                              : "Result incorrect!\n");

        // All resources (host vectors and device memory) are released
        // automatically when they go out of scope.
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

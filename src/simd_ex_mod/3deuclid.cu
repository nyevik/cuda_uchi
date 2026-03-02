/********************************************************************
 *  Euclidean distance between two 3‑D point arrays (CUDA version)
 *
 *  * Point3D – simple POD with a static distance helper.
 *  * DeviceBuffer – RAII wrapper for device allocations.
 *  * calculateEuclideanDistanceKernel – free‑standing __global__ kernel.
 *  * host driver (main) demonstrates allocation, launch, and verification.
 ********************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------
 *  Simple error‑checking macro – throws on failure.
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
 *  POD‑like point type – public members for easy kernel access.
 * ------------------------------------------------------------------ */
struct Point3D
{
    float x, y, z;

    // Defaulted special members – they are trivially copyable.
    __host__ __device__ Point3D() = default;
    __host__ __device__ Point3D(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

    // Static distance helper usable from host and device.
    __host__ __device__ static inline float distance(const Point3D &a,
                                                     const Point3D &b)
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }
};

/* ------------------------------------------------------------------
 *  RAII wrapper for device memory.
 * ------------------------------------------------------------------ */
template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count = 0) : count_(count), devPtr_(nullptr)
    {
        if (count_ > 0)
        {
            CUDA_CHECK(cudaMalloc(&devPtr_, count_ * sizeof(T)));
        }
    }

    // non‑copyable
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    // movable
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : devPtr_(other.devPtr_), count_(other.count_)
    {
        other.devPtr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            devPtr_ = other.devPtr_;
            count_ = other.count_;
            other.devPtr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    // Accessors
    T *data() noexcept { return devPtr_; }
    const T *data() const noexcept { return devPtr_; }
    std::size_t size() const noexcept { return count_; }

private:
    void release() noexcept
    {
        if (devPtr_)
        {
            cudaFree(devPtr_);
            devPtr_ = nullptr;
        }
    }

    T *devPtr_;
    std::size_t count_;
};

/* ------------------------------------------------------------------
 *  Kernel – computes Euclidean distance for each pair of points.
 * ------------------------------------------------------------------ */
__global__ void calculateEuclideanDistanceKernel(const Point3D *__restrict__ lineA,
                                                 const Point3D *__restrict__ lineB,
                                                 float *__restrict__ distances,
                                                 int numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        distances[idx] = Point3D::distance(lineA[idx], lineB[idx]);
    }
}

/* ------------------------------------------------------------------
 *  Host helper – fills a vector with deterministic data.
 * ------------------------------------------------------------------ */
static void fill_points(std::vector<Point3D> &vec, float offset)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = Point3D(static_cast<float>(i) + offset,
                         static_cast<float>(i) * 0.5f + offset,
                         static_cast<float>(i) * 0.25f + offset);
    }
}

/* ------------------------------------------------------------------
 *  Main driver – allocate, launch, verify.
 * ------------------------------------------------------------------ */
int main()
{
    try
    {
        const int numPoints = 1'000'000; // 1 M points
        const size_t bytes = numPoints * sizeof(Point3D);
        const size_t outBytes = numPoints * sizeof(float);

        // ---------- Host data ----------
        std::vector<Point3D> h_A(numPoints);
        std::vector<Point3D> h_B(numPoints);
        std::vector<float> h_distances(numPoints, 0.0f);

        fill_points(h_A, 0.0f);
        fill_points(h_B, 5.0f); // shift B so distances are non‑zero

        // ---------- Device buffers ----------
        DeviceBuffer<Point3D> d_A(numPoints);
        DeviceBuffer<Point3D> d_B(numPoints);
        DeviceBuffer<float> d_dist(numPoints);

        // ---------- Copy host → device ----------
        CUDA_CHECK(cudaMemcpy(d_A.data(), h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.data(), h_B.data(), bytes, cudaMemcpyHostToDevice));

        // ---------- Kernel launch ----------
        const int threadsPerBlock = 256;
        const int blocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

        calculateEuclideanDistanceKernel<<<blocks, threadsPerBlock>>>(
            d_A.data(), d_B.data(), d_dist.data(), numPoints);
        CUDA_CHECK(cudaGetLastError()); // launch errors
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------- Copy result back ----------
        CUDA_CHECK(cudaMemcpy(h_distances.data(),
                              d_dist.data(),
                              outBytes,
                              cudaMemcpyDeviceToHost));

        // ---------- Simple verification ----------
        bool ok = true;
        for (int i = 0; i < numPoints; ++i)
        {
            float dx = h_A[i].x - h_B[i].x;
            float dy = h_A[i].y - h_B[i].y;
            float dz = h_A[i].z - h_B[i].z;
            float ref = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (std::abs(ref - h_distances[i]) > 1e-5f)
            {
                std::cerr << "Mismatch at " << i << ": host=" << ref
                          << " gpu=" << h_distances[i] << '\n';
                ok = false;
                break;
            }
        }
        std::cout << (ok ? "All distances match.\n" : "Verification failed.\n");

        // No explicit cudaFree – DeviceBuffer destructor does it.
        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
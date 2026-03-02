/** Euclidean distance between two points in 3D space */

#include <iostream>

class Point3D
{
public:
    // Constructor
    __host__ __device__ Point3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    // Destructor
    __host__ __device__ ~Point3D() {}

    // Copy constructor
    __host__ __device__ Point3D(const Point3D &other) : x(other.x), y(other.y), z(other.z) {}
    // Copy Assignment operator
    __host__ __device__ Point3D &operator=(const Point3D &other)
    {
        if (this != &other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }

    // Move constructor
    __host__ __device__ Point3D(Point3D &&other) noexcept : x(other.x), y(other.y), z(other.z)
    {
        other.x = 0;
        other.y = 0;
        other.z = 0;
    }
    // Move Assignment operator
    __host__ __device__ Point3D &operator=(Point3D &&other) noexcept
    {
        if (this != &other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
            other.x = 0;
            other.y = 0;
            other.z = 0;
        }
        return *this;
    }
    // Destructor
    __host__ __device__ ~Point3D() {}

private:
    float x, y, z; // Coordinates of the point
};

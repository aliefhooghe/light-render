#ifndef FLOAT3_OPERATORS_CUH
#define FLOAT3_OPERATORS_CUH

#include <cuda.h>

namespace Xrender {

    static __device__ __host__ __forceinline__ float3 operator + (const float3& a, const float3& b)
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    static __device__ __host__ __forceinline__ float3& operator += (float3& a, const float3& b)
    {
        a = a + b;
        return a;
    }

    static __device__ __host__ __forceinline__ float3 operator - (const float3& a, const float3& b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    static __device__ __host__ __forceinline__ float3& operator -= (float3& a, const float3& b)
    {
        a = a - b;
        return a;
    }

    static __device__ __host__ __forceinline__ float3 operator -(const float3& a)
    {
        return {-a.x, -a.y, -a.z};
    }

    static __device__ __host__ __forceinline__ float3 operator * (const float3& a, const float3& b)
    {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }

    static __device__ __host__ __forceinline__ float3 operator * (const float& a, const float3& b)
    {
        return {a * b.x, a * b.y, a * b.z};
    }

    static __device__ __host__ __forceinline__ float3 operator * (const float3& b, const float& a)
    {
        return {b.x * a, b.y * a, b.z * a};
    }

    static __device__ __host__ __forceinline__ float3& operator *= (float3& a, const float& b)
    {
        a = a * b;
        return a;
    }

    static __device__ __host__ __forceinline__ float3& operator *= (float3& a, const float3& b)
    {
        a = a * b;
        return a;
    }

    static __device__ __host__ __forceinline__ float3 operator / (const float3& a, const float3& b)
    {
        return {a.x / b.x, a.y / b.y, a.z / b.z};
    }

    static __device__ __host__ __forceinline__ float3 operator / (const float& a, const float3& b)
    {
        return {a / b.x, a / b.y, a / b.z};
    }

    static __device__ __host__ __forceinline__ float3 operator / (const float3& b, const float& a)
    {
        return {b.x / a, b.y / a, b.z / a};
    }

    static __device__ __host__ __forceinline__ float3& operator /= (float3& a, const float& b)
    {
        a = a / b;
        return a;
    }

    static __device__ __host__ __forceinline__ float3& operator /= (float3& a, const float3& b)
    {
        a = a / b;
        return a;
    }

    static __device__ __host__ __forceinline__ float dot(const float3& a, const float3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static __device__ __host__ __forceinline__ float norm2(const float3& a)
    {
        return dot(a, a);
    }

    static __device__ __host__ __forceinline__ float norm(const float3& a)
    {
        return __sqrtf(norm2(a));
    }

    static __device__ __host__ __forceinline__ float3 normalized(const float3& a)
    {
        return rsqrtf(norm2(a)) * a;
    }

    static __device__ __host__ __forceinline__ void normalize(float3& a)
    {
        a *= rsqrtf(norm2(a));
    }

    static __device__ __host__ __forceinline__ float3 cross(const float3 &x, const float3 &y)
    {
        return {x.y * y.z - x.z * y.y,
                x.z * y.x - x.x * y.z,
                x.x * y.y - x.y * y.x};
    }

    static __device__ __host__ __forceinline__ float3 min(const float3 &x, const float3 &y)
    {
        return {
            x.x < y.x ? x.x : y.x,
            x.y < y.y ? x.y : y.y,
            x.z < y.z ? x.z : y.z
        };
    }

    static __device__ __host__ __forceinline__ float3 max(const float3 &x, const float3 &y)
    {
        return {
            x.x > y.x ? x.x : y.x,
            x.y > y.y ? x.y : y.y,
            x.z > y.z ? x.z : y.z
        };
    }

    static __device__ __host__ __forceinline__ float& dim_val(float3& x, unsigned int dim)
    {
        return reinterpret_cast<float*>(&x)[dim];
    }

    static __device__ __host__ __forceinline__ const float& dim_val(const float3& x, unsigned int dim)
    {
        return reinterpret_cast<const float*>(&x)[dim];
    }

}

#endif
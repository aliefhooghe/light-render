#ifndef VEC_H_
#define VEC_H_

#include <utility>
#include <cmath>
#include <algorithm>
#include <ostream>

namespace Xrender {

    /**
     * \brief vect represent a point or a vector in the 3D space
     */
    template <typename T>
    struct vect {

        constexpr vect operator+(const vect& other) const noexcept
        {
            return {x + other.x, y + other.y, z + other.z};
        }

        constexpr vect& operator+=(const vect& other) noexcept
        {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        constexpr vect operator-(const vect &other) const noexcept
        {
            return {x - other.x, y - other.y, z - other.z};
        }

        constexpr vect operator-=(const vect &other) noexcept
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        constexpr vect operator*(T a) const noexcept
        {
            return {a * x, a * y, a * z};
        }

        constexpr vect operator*=(T a) noexcept
        {
            x *= a;
            y *= a;
            z *= a;
            return *this;
        }

        constexpr vect operator/(T a) const noexcept
        {
            return {x / a, y / a, z / a};
        }

        constexpr vect operator/=(T a) noexcept
        {
            x /= a;
            y /= a;
            z /= a;
            return *this;
        }

        constexpr T norm2() const noexcept
        {
            return x*x + y*y + z*z;
        }

        constexpr T norm() const noexcept
        {
            return std::sqrt(norm2());
        }

        void normalize() noexcept
        {
            const T n = norm();
            x /= n;
            y /= n;
            z /= n;
        }

        vect normalized() const noexcept
        {
            const T n = norm();
            return {x/n, y/n, z/n};
        }

        T x, y, z;
    };

    template <typename T>
    vect<T> operator*(T a, const vect<T>& v) noexcept
    {
        return {a * v.x, a * v.y, a * v.z}; 
    }

    template <typename T>
    vect<T> operator-(const vect<T>& v)
    {
        return {-v.x, -v.y, -v.z};
    }

    template <typename T>
    T dot(const vect<T> &x, const vect<T> &y) noexcept
    {
        return x.x * y.x + x.y * y.y + x.z * y.z;
    }

    template <typename T>
    vect<T> cross(const vect<T> &x, const vect<T> &y) noexcept
    {
        return {x.y * y.z - x.z * y.y,
                x.z * y.x - x.x * y.z,
                x.x * y.y - x.y * y.x};
    }

    template <typename T>
    T distance(const vect<T> &x, const vect<T> &y) noexcept
    {
        return (x - y).norm();
    }

    template <typename T>
    T distance2(const vect<T> &x, const vect<T> &y) noexcept
    {
        return (x - y).norm2();
    }

    template <typename T> 
    vect<T> unit_dir(const vect<T>& from, const vect<T>& to) noexcept
    {
        return (to - from).normalized();
    }

    template <typename T>
    vect<T> min(const vect<T>& v1, const vect<T>& v2) noexcept
    {
        return {
            std::min(v1.x, v2.x),
            std::min(v1.y, v2.y),
            std::min(v1.z, v2.z)
        };
    }

    template <typename T>
    vect<T> max(const vect<T>& v1, const vect<T>& v2) noexcept
    {
        return {
            std::max(v1.x, v2.x),
            std::max(v1.y, v2.y),
            std::max(v1.z, v2.z)
        };
    }

    template <typename T>
    auto& operator<<(std::ostream& stream, const vect<T>& v)
    {
        return stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }

    using vecf = vect<float>;
}

#endif
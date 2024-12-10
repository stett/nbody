#pragma once
#include <cmath>
#include <limits>

namespace nbody
{
    struct Vector;
    Vector operator*(float s, Vector v);
    float dot(const Vector& a, const Vector& b);
    Vector cross(const Vector& a, const Vector& b);

    struct Vector
    {
        float x = 0;
        float y = 0;
        float z = 0;

        float& operator[](const size_t i) {
            return *((&x) + i);
        }

        const float& operator[](const size_t i) const {
            return *((&x) + i);
        }

        float size_sq() const
        {
            return dot(*this, *this);
        }

        float size() const
        {
            return std::sqrt(size_sq());
        }

        float size_inv(const float val_if_inf = 0.f) const
        {
            const float _size_sq = size_sq();
            return _size_sq > std::numeric_limits<float>::epsilon()
                ? 1.f / std::sqrt(_size_sq)
                : val_if_inf;
        }

        float dist_sq(const Vector& other) const
        {
            const Vector delta = other - *this;
            return dot(delta, delta);
        }

        Vector norm() const
        {
            return (*this)/size();
        }

        bool operator==(const Vector& other) const = default;
        Vector operator+(const Vector& other) const { return { x + other.x, y + other.y, z + other.z }; }
        Vector operator-(const Vector& other) const { return { x - other.x, y - other.y, z - other.z }; }
        Vector operator*(float s) const { return { x * s, y * s, z * s }; }
        Vector operator/(float s) const { return { x / s, y / s, z / s }; }
        Vector operator*(const Vector& other) const { return { x * other.x, y * other.y, z * other.z }; }
        Vector& operator+=(const Vector& other) { x += other.x; y += other.y; z += other.z; return *this; }
        Vector& operator-=(const Vector& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    };
}
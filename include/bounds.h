#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include "vector.h"
#include "ray.h"

namespace nbody
{
    struct Bounds
    {
        float size;
        Vector center;

        [[nodiscard]]
        Vector min() const
        {
            const float half = size * .5f;
            return { center.x - half, center.y - half, center.z - half };
        }

        [[nodiscard]]
        Vector max() const
        {
            const float half = size * .5f;
            return { center.x + half, center.y + half, center.z + half };
        }

        [[nodiscard]]
        uint8_t quadrant(const Vector& pos) const
        {
            return
                ((pos.x < center.x) << 0) |
                ((pos.y < center.y) << 1) |
                ((pos.z < center.z) << 2);
        }

        [[nodiscard]]
        Bounds quadrant_bounds(const uint8_t q) const
        {
            const float half = size * .5f;
            const float quart = size * .25f;
            return {
                .size = half,
                .center = {
                    .x = ((q & (1 << 0)) != 0) ? (center.x - quart) : (center.x + quart),
                    .y = ((q & (1 << 1)) != 0) ? (center.y - quart) : (center.y + quart),
                    .z = ((q & (1 << 2)) != 0) ? (center.z - quart) : (center.z + quart),
                }
            };
        }

        [[nodiscard]]
        bool contains(const Vector& pos) const
        {
            const float half = size * .5f;
            return
                center.x - half <= pos.x && pos.x <= center.x + half &&
                center.y - half <= pos.y && pos.y <= center.y + half &&
                center.z - half <= pos.z && pos.z <= center.z + half;
        }

        bool ray_intersect(const Ray& ray, Vector& out_point, float& out_length) const
        {
            // If the origin is inside the bounds, it's an intersection
            bool origin_inside = true;
            for (size_t i = 0; i < 3; ++i)
                if (ray.origin[i] < center[i] - size || ray.origin[i] > center[i] + size)
                { origin_inside = false; break; }
            if (origin_inside)
            {
                out_point = ray.origin;
                return true;
            }

            // Test right & left faces
            if (ray_intersect_coord<0>(ray, out_point, out_length) ||
                ray_intersect_coord<1>(ray, out_point, out_length) ||
                ray_intersect_coord<2>(ray, out_point, out_length))
                return true;

            return false;
        }

        [[nodiscard]]
        bool ray_intersect(const Ray& ray) const
        {
            Vector point;
            float length = std::numeric_limits<float>::max();
            return ray_intersect(ray, point, length);
        }

    private:

        template <size_t coord>
        bool ray_intersect_coord(const Ray& ray, Vector& point, float& length) const
        {
            static constexpr size_t i = coord;
            static constexpr size_t j = (i + 1) % 3;
            static constexpr size_t k = (i + 2) % 3;

            if (ray.direction[i] < -std::numeric_limits<float>::epsilon())
            {
                float t = (center[i] + size - ray.origin[i]) / ray.direction[i];
                if (t < length)
                {
                    Vector p = ray.origin + (ray.direction * t);
                    if (p[j] > center[j] - size && p[j] < center[j] + size && p[k] > center[k] - size && p[k] < center[k] + size)
                    {
                        point = p;
                        length = t;
                        return true;
                    }
                }
            }
            else if (ray.direction[i] > std::numeric_limits<float>::epsilon())
            {
                float t = (center[i] - size - ray.origin[i]) / ray.direction[i];
                if (t < length)
                {
                    Vector p = ray.origin + (ray.direction * t);
                    if (p[j] > center[j] - size && p[j] < center[j] + size && p[k] > center[k] - size && p[k] < center[k] + size)
                    {
                        point = p;
                        length = t;
                        return true;
                    }
                }
            }
            else
            {
                return false;
            }
        }
    };
}

#pragma once

namespace nbody::detail
{
    template <typename T>
    struct uninitialized_allocator : std::allocator<T> {
        template <typename U> struct rebind { using other = uninitialized_allocator<U>; };
        template <typename U> void construct(U*) noexcept {} // Do nothing
        template <typename U, typename... Args> void construct(U* p, Args&&... args) {
            ::new((void*)p) U(std::forward<Args>(args)...);
        }
    };
}
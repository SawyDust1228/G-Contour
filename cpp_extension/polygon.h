#pragma once

#include <assert.h>
#include <vector>

template <typename T>
struct PointCPU
{
    PointCPU() : x(0), y(0) {}
    PointCPU(T x, T y) : x(x), y(y) {}

    T x;
    T y;
};

template <typename T>
struct polygon_cpu
{
    void addPoint(const PointCPU<T>& p) { polygon.push_back(p); }

    PointCPU<T>& operator[](int idx)
    {
        assert(idx >= 0 && idx < getNumPoints());
        return polygon[idx];
    }

    int getNumPoints() const { return polygon.size(); }

    std::vector<PointCPU<T>> polygon;
};

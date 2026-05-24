#pragma once

#include <cstdlib>
#include <iostream>

#include <torch/extension.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous")
#define CHECK_CPU(x)                                                           \
    TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")

#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#define CUDA_ASSERT(condition)                                                 \
    if (!(condition))                                                          \
    {                                                                          \
        return;                                                                \
    }

namespace
{

bool is_debug_enabled() { return std::getenv("GCONTOUR_DEBUG") != nullptr; }

void debug_start(const char* stage)
{
    if (is_debug_enabled())
    {
        std::cerr << "[gcontour] start " << stage << std::endl;
    }
}

void debug_sync(const char* stage)
{
#ifdef __CUDACC__
    if (!is_debug_enabled())
    {
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "[gcontour] ", stage,
                " failed: ", cudaGetErrorString(err));
    std::cerr << "[gcontour] done " << stage << std::endl;
#else
    (void)stage;
#endif
}

} // namespace

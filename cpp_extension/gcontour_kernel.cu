
#include <cassert>
#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

#include "ATen/TensorIndexing.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/ops/_unique.h"
#include "check.h"

__device__ const int dirx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
__device__ const int diry[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

#define threshold 0.5

__device__ bool inRange(int x, int y, int height, int width)
{
    return x >= 0 && x < height and y >= 0 && y < width;
}

__device__ bool
isboundary(const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
               label_tensor,
           int x, int y)
{

    for (int k = 1; k < 8; k = k + 2)
    {
        int id = x + dirx[k];
        int jd = y + diry[k];

        if (!inRange(id, jd, label_tensor.size(0), label_tensor.size(1)))
        {
            return true;
        }
        if (label_tensor[id][jd] < threshold)
            return true;
    }
    return false;
}

__device__ bool isIsolatePixel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    int x, int y)
{
    int counter = 0;
    for (int k = 0; k < 8; k++)
    {
        int id = x + dirx[k];
        int jd = y + diry[k];

        if (!inRange(id, jd, label_tensor.size(0), label_tensor.size(1)))
        {
            continue;
        }
        if (label_tensor[id][jd] < threshold)
        {
            continue;
        }

        counter += 1;
    }

    return counter <= 1;
}

__global__ void contour_tracing_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> image,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> contour)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < image.size(0) && idy < image.size(1))
    {
        if (image[idx][idy] && isboundary(image, idx, idy))
        {
            if (!isIsolatePixel(image, idx, idy))
            {
                contour[idx][idy] = 1;
            }
            else
            {
                contour[idx][idy] = 0;
            }
        }
        else
        {
            contour[idx][idy] = 0;
        }
    }
}

torch::Tensor get_contour(torch::Tensor image)
{
    CHECK_CUDA(image);
    CHECK_CONTIGUOUS(image);

    image = image.to(torch::kInt);

    cudaSetDevice(image.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor contour = torch::zeros(
        image.sizes(), torch::dtype(torch::kInt).device(image.device()));

    int thd_x = 32;
    int thd_y = 32;
    dim3 block(thd_x, thd_y);
    dim3 grid((image.size(0) + thd_x - 1) / thd_x,
              (image.size(1) + thd_y - 1) / thd_y);

    contour_tracing_kernel<<<grid, block, 0, stream>>>(
        image.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        contour.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    return contour;
}

__device__ int uf_find_root(
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    int label, int width)
{
    int parent = label;
    int guard = 0;
    while (guard++ < labels.size(0) * labels.size(1))
    {
        int px = (parent - 1) / width;
        int py = (parent - 1) % width;
        int grand = labels[px][py];
        if (grand == parent)
        {
            break;
        }
        parent = grand;
    }
    return parent;
}

__device__ int
uf_find(torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
        int label, int width)
{
    int parent = uf_find_root(labels, label, width);

    int cur = label;
    int guard = 0;
    while (cur != parent && guard++ < labels.size(0) * labels.size(1))
    {
        int cx = (cur - 1) / width;
        int cy = (cur - 1) % width;
        int next = labels[cx][cy];
        labels[cx][cy] = parent;
        cur = next;
    }
    return parent;
}

__device__ void
uf_union(torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
         int a, int b, int width)
{
    while (true)
    {
        int root_a = uf_find_root(labels, a, width);
        int root_b = uf_find_root(labels, b, width);
        if (root_a == root_b)
        {
            return;
        }

        int high = max(root_a, root_b);
        int low = min(root_a, root_b);
        int hx = (high - 1) / width;
        int hy = (high - 1) % width;
        int old = atomicMin(&labels[hx][hy], low);
        if (old == high || old == low)
        {
            return;
        }
    }
}

__global__ void init_labels_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> image,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= image.size(0) || idy >= image.size(1))
    {
        return;
    }
    labels[idx][idy] = image[idx][idy] ? (idx * image.size(1) + idy + 1) : 0;
}

__global__ void union_labels_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> image,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= image.size(0) || idy >= image.size(1) || labels[idx][idy] == 0)
    {
        return;
    }

    const int h = image.size(0);
    const int w = image.size(1);
    const int label = labels[idx][idy];
    for (int k = 0; k < 8; ++k)
    {
        int nx = idx + dirx[k];
        int ny = idy + diry[k];
        if (inRange(nx, ny, h, w) && labels[nx][ny] != 0)
        {
            uf_union(labels, label, labels[nx][ny], w);
        }
    }
}

__global__ void compress_labels_kernel(
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= labels.size(0) || idy >= labels.size(1) || labels[idx][idy] == 0)
    {
        return;
    }
    labels[idx][idy] = uf_find(labels, labels[idx][idy], labels.size(1));
}

torch::Tensor getLabelTensor(torch::Tensor image)
{
    CHECK_CUDA(image);
    CHECK_CONTIGUOUS(image);

    image = image.to(torch::kInt);
    cudaSetDevice(image.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor label_tensor = torch::zeros(
        image.sizes(), torch::dtype(torch::kInt).device(image.device()));

    int thd_x = 32;
    int thd_y = 32;
    dim3 block(thd_x, thd_y);
    dim3 grid((image.size(0) + thd_x - 1) / thd_x,
              (image.size(1) + thd_y - 1) / thd_y);

    init_labels_kernel<<<grid, block, 0, stream>>>(
        image.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
    union_labels_kernel<<<grid, block, 0, stream>>>(
        image.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
    compress_labels_kernel<<<grid, block, 0, stream>>>(
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    return label_tensor;
}

torch::Tensor getUniqueValues(torch::Tensor label_tensor)
{
    CHECK_CUDA(label_tensor);
    CHECK_CONTIGUOUS(label_tensor);
    using namespace torch::indexing;
    auto unique_values = std::get<0>(torch::_unique(label_tensor));
    assert(unique_values[0].item().toInt() == 0);
    unique_values = unique_values.index({Slice(1, None)});
    return unique_values;
}

__global__ void build_label_index_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        unique,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> label_index)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < unique.size(0))
    {
        label_index[unique[idx]] = idx;
    }
}

__global__ void get_start_points_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        label_index,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> min_keys)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= label_tensor.size(0) || idy >= label_tensor.size(1) ||
        !contour[idx][idy])
    {
        return;
    }

    int label = label_tensor[idx][idy];
    if (label == 0)
    {
        return;
    }

    bool has_opposite_pair = false;
    for (int k = 0; k < 4; ++k)
    {
        int x0 = idx + dirx[k];
        int y0 = idy + diry[k];
        int x1 = idx + dirx[k + 4];
        int y1 = idy + diry[k + 4];
        if (inRange(x0, y0, label_tensor.size(0), label_tensor.size(1)) &&
            inRange(x1, y1, label_tensor.size(0), label_tensor.size(1)) &&
            label_tensor[x0][y0] == label && label_tensor[x1][y1] == label)
        {
            has_opposite_pair = true;
            break;
        }
    }
    if (has_opposite_pair)
    {
        return;
    }

    int component_idx = label_index[label];
    CUDA_ASSERT(component_idx >= 0);

    int key = idy * label_tensor.size(0) + idx;
    atomicMin(&min_keys[component_idx], key);
}

__global__ void decode_start_points_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        min_keys,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> points,
    int height)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < min_keys.size(0))
    {
        int key = min_keys[idx];
        if (key == INT_MAX)
        {
            points[idx][0] = -1;
            points[idx][1] = -1;
        }
        else
        {
            points[idx][0] = key % height;
            points[idx][1] = key / height;
        }
    }
}

torch::Tensor get_start_points(torch::Tensor contour,
                               torch::Tensor label_tensor,
                               torch::Tensor unique_tensor)
{
    CHECK_CUDA(contour);
    CHECK_CONTIGUOUS(contour);
    CHECK_CUDA(label_tensor);
    CHECK_CONTIGUOUS(label_tensor);

    CHECK_CUDA(unique_tensor);
    CHECK_CONTIGUOUS(unique_tensor);

    cudaSetDevice(label_tensor.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int n = unique_tensor.size(0);
    const int label_count = label_tensor.numel() + 1;

    torch::Tensor label_index =
        torch::full({label_count}, -1,
                    torch::dtype(torch::kInt).device(label_tensor.device()));
    torch::Tensor min_keys = torch::full(
        {n}, INT_MAX, torch::dtype(torch::kInt).device(label_tensor.device()));
    torch::Tensor points = torch::empty(
        {n, 2}, torch::dtype(torch::kInt).device(label_tensor.device()));

    const int one_d_threads = 256;
    const int one_d_blocks = (n + one_d_threads - 1) / one_d_threads;
    build_label_index_kernel<<<one_d_blocks, one_d_threads, 0, stream>>>(
        unique_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        label_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    int thd_x = 32;
    int thd_y = 32;
    dim3 block(thd_x, thd_y);
    dim3 grid((label_tensor.size(0) + thd_x - 1) / thd_x,
              (label_tensor.size(1) + thd_y - 1) / thd_y);

    get_start_points_kernel<<<grid, block, 0, stream>>>(
        contour.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        min_keys.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    decode_start_points_kernel<<<one_d_blocks, one_d_threads, 0, stream>>>(
        min_keys.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        points.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.size(0));

    return points;
}

__global__ void count_contour_pixels_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        label_index,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> counts)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= contour.size(0) || idy >= contour.size(1) || !contour[idx][idy])
    {
        return;
    }
    int label = label_tensor[idx][idy];
    int component_idx = label_index[label];
    CUDA_ASSERT(component_idx >= 0);
    atomicAdd(&counts[component_idx], 1);
}

__device__ bool contour_walk_count(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    int label, int startx, int starty, int initial_orient, int max_steps,
    int& vertex_count, int& step_count)
{
    const int h = contour.size(0), w = contour.size(1);
    int orient = initial_orient;
    int nowx = startx, nowy = starty;
    int nextx = -1, nexty = -1;
    vertex_count = 1;
    step_count = 0;

    while ((nextx != startx || nexty != starty) && step_count++ < max_steps)
    {
        nextx = -1;
        nexty = -1;
        int nextorient = -1;
        for (int k = -2; k < 6; ++k)
        {
            int candidate_orient = (orient + k + 8) % 8;
            int newx = nowx + dirx[candidate_orient];
            int newy = nowy + diry[candidate_orient];
            if (inRange(newx, newy, h, w) && contour[newx][newy] &&
                label_tensor[newx][newy] == label)
            {
                nextx = newx;
                nexty = newy;
                nextorient = candidate_orient;
                break;
            }
        }
        if (nextx < 0)
        {
            return false;
        }
        if (nextorient != orient && !(nowx == startx && nowy == starty))
        {
            ++vertex_count;
        }
        nowx = nextx;
        nowy = nexty;
        orient = nextorient;
    }

    return nowx == startx && nowy == starty && step_count <= max_steps;
}

__device__ int choose_initial_orient(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    int label, int startx, int starty, int max_steps, int& best_vertices)
{
    best_vertices = 0;
    int best_orient = -1;
    for (int orient = 0; orient < 8; ++orient)
    {
        int nx = startx + dirx[orient];
        int ny = starty + diry[orient];
        if (!inRange(nx, ny, contour.size(0), contour.size(1)) ||
            !contour[nx][ny] || label_tensor[nx][ny] != label)
        {
            continue;
        }
        int vertices = 0;
        int steps = 0;
        if (contour_walk_count(contour, label_tensor, label, startx, starty,
                               orient, max_steps, vertices, steps) &&
            vertices > best_vertices)
        {
            best_vertices = vertices;
            best_orient = orient;
        }
    }
    return best_orient;
}

__global__ void get_shape_by_walk_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        unique,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        start_points,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        contour_counts,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        shape_tensor,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        orient_tensor)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= unique.size(0))
    {
        return;
    }

    const int startx = start_points[idx][0];
    const int starty = start_points[idx][1];
    if (startx < 0 || starty < 0)
    {
        shape_tensor[idx] = 0;
        orient_tensor[idx] = -1;
        return;
    }

    const int label = unique[idx];
    const int max_steps = contour_counts[idx] * 4 + 8;
    int vertices = 0;
    int orient = choose_initial_orient(contour, label_tensor, label, startx,
                                       starty, max_steps, vertices);
    if (orient < 0 || vertices < 3)
    {
        shape_tensor[idx] = 0;
        orient_tensor[idx] = -1;
        return;
    }

    shape_tensor[idx] = vertices;
    orient_tensor[idx] = orient;
}

__global__ void make_max_steps_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        contour_counts,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> max_steps)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < contour_counts.size(0))
    {
        max_steps[idx] = contour_counts[idx] * 4 + 8;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
get_shape_capacity(torch::Tensor unique, torch::Tensor contour,
                   torch::Tensor label_tensor, torch::Tensor start_points)
{

    CHECK_CUDA(label_tensor);
    CHECK_CONTIGUOUS(label_tensor);
    CHECK_CUDA(unique);
    CHECK_CONTIGUOUS(unique);
    CHECK_CUDA(contour);
    CHECK_CONTIGUOUS(contour);
    CHECK_CUDA(start_points);
    CHECK_CONTIGUOUS(start_points);

    cudaSetDevice(label_tensor.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int n = unique.size(0);
    const int label_count = label_tensor.numel() + 1;

    torch::Tensor shape_tensor =
        torch::zeros({n}, torch::dtype(torch::kInt).device(contour.device()));
    torch::Tensor orient_tensor = torch::full(
        {n}, -1, torch::dtype(torch::kInt).device(contour.device()));
    torch::Tensor contour_counts =
        torch::zeros({n}, torch::dtype(torch::kInt).device(contour.device()));
    torch::Tensor max_steps_tensor =
        torch::zeros({n}, torch::dtype(torch::kInt).device(contour.device()));
    torch::Tensor label_index =
        torch::full({label_count}, -1,
                    torch::dtype(torch::kInt).device(label_tensor.device()));

    const int one_d_threads = 256;
    const int one_d_blocks = (n + one_d_threads - 1) / one_d_threads;
    build_label_index_kernel<<<one_d_blocks, one_d_threads, 0, stream>>>(
        unique.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        label_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    int thd_x = 32;
    int thd_y = 32;
    dim3 block2d(thd_x, thd_y);
    dim3 grid2d((contour.size(0) + thd_x - 1) / thd_x,
                (contour.size(1) + thd_y - 1) / thd_y);
    count_contour_pixels_kernel<<<grid2d, block2d, 0, stream>>>(
        contour.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        contour_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>());
    make_max_steps_kernel<<<one_d_blocks, one_d_threads, 0, stream>>>(
        contour_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        max_steps_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;

    get_shape_by_walk_kernel<<<blocks, threads, 0, stream>>>(
        unique.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        contour.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        start_points.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        contour_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        shape_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        orient_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>());

    return {shape_tensor, start_points, orient_tensor, max_steps_tensor};
}

__global__ void trace_contour_kernel(
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        min_points,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        orient_tensor,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        max_steps_tensor,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        unique,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        contour,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        image,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        label_tensor,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ptr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        real_shape_tensor,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> polygons)
{

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < unique.size(0))
    {
        const int h = contour.size(0), w = contour.size(1);
        const int offset = ptr[idx];
        const int capacity = ptr[idx + 1] - ptr[idx];
        if (capacity <= 0)
        {
            return;
        }
        int label = unique[idx];

        int startx = min_points[idx][0];
        int starty = min_points[idx][1];
        int orient = orient_tensor[idx];
        if (startx < 0 || starty < 0 || orient < 0)
        {
            real_shape_tensor[idx] = 0;
            return;
        }
        int nowx = startx, nowy = starty;
        int cnt_c = 0;
        polygons[offset + cnt_c][0] = nowx;
        polygons[offset + cnt_c][1] = nowy;
        cnt_c++;
        int nextx = -1, nexty = -1;
        int clockwise = -1;
        int guard = 0;
        int max_steps = max_steps_tensor[idx];
        bool closed = false;
        while ((nextx != startx || nexty != starty) && guard++ < max_steps)
        {
            nextx = -1;
            nexty = -1;
            int nextorient = -2;
            for (int k = -2; k < 6; k++)
            {
                int newx = nowx + dirx[(orient + k + 8) % 8];
                int newy = nowy + diry[(orient + k + 8) % 8];
                if (inRange(newx, newy, h, w))
                {
                    if (contour[newx][newy] &&
                        label_tensor[newx][newy] == label)
                    {
                        nextx = newx;
                        nexty = newy;
                        nextorient = (orient + k + 8) % 8;
                        break;
                    }
                    if (image[newx][newy])
                    {
                        clockwise = 1;
                    }
                }
            }
            if (nextx < 0)
                break;
            if (nextorient != orient && !(nowx == startx && nowy == starty))
            {
                if (cnt_c < capacity)
                {
                    polygons[offset + cnt_c][0] = nowx;
                    polygons[offset + cnt_c][1] = nowy;
                    cnt_c++;
                }
            }
            nowx = nextx;
            nowy = nexty;
            orient = nextorient;
            closed = nowx == startx && nowy == starty;
        }

        if (!closed)
        {
            real_shape_tensor[idx] = 0;
            return;
        }

        if (cnt_c > 1 && polygons[offset][0] == polygons[offset + 1][0] &&
            polygons[offset][1] == polygons[offset + 1][1])
        {
            cnt_c--;
            for (int i = 0; i < cnt_c; i++)
            {
                polygons[offset + i][0] = polygons[offset + i + 1][0];
                polygons[offset + i][1] = polygons[offset + i + 1][1];
            }
        }
        if (clockwise == -1)
        {
            for (int i = 0; i < (cnt_c + 1) / 2; i++)
            {
                int a = polygons[offset + cnt_c - 1 - i][0];
                polygons[offset + cnt_c - 1 - i][0] = polygons[offset + i][0];
                polygons[offset + i][0] = a;
                a = polygons[offset + cnt_c - 1 - i][1];
                polygons[offset + cnt_c - 1 - i][1] = polygons[offset + i][1];
                polygons[offset + i][1] = a;
            }
        }

        real_shape_tensor[idx] = cnt_c;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
trace_contour(torch::Tensor shape_tensor, torch::Tensor min_points,
              torch::Tensor orient_tensor, torch::Tensor max_steps_tensor,
              torch::Tensor unique, torch::Tensor contour, torch::Tensor image,
              torch::Tensor label_tensor)
{
    using namespace torch::indexing;

    CHECK_CUDA(label_tensor);
    CHECK_CONTIGUOUS(label_tensor);
    CHECK_CUDA(unique);
    CHECK_CONTIGUOUS(unique);
    CHECK_CUDA(contour);
    CHECK_CONTIGUOUS(contour);
    CHECK_CUDA(min_points);
    CHECK_CONTIGUOUS(min_points);
    CHECK_CUDA(image);
    CHECK_CONTIGUOUS(image);
    CHECK_CUDA(shape_tensor);
    CHECK_CONTIGUOUS(shape_tensor);
    CHECK_CUDA(orient_tensor);
    CHECK_CONTIGUOUS(orient_tensor);
    CHECK_CUDA(max_steps_tensor);
    CHECK_CONTIGUOUS(max_steps_tensor);

    cudaSetDevice(image.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int total_num = shape_tensor.sum().item().toInt();
    torch::Tensor ptr =
        torch::zeros({1 + shape_tensor.size(0)},
                     torch::dtype(torch::kInt).device(image.device()));
    ptr.index({Slice(1, None)}) += shape_tensor;
    ptr = ptr.cumsum(0).to(torch::kInt);

    const int n = unique.size(0);

    torch::Tensor polygons = torch::zeros(
        {total_num, 2}, torch::dtype(torch::kInt).device(image.device()));

    torch::Tensor real_shape_tensor =
        torch::zeros({n}, torch::dtype(torch::kInt).device(image.device()));

    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;

    trace_contour_kernel<<<blocks, threads, 0, stream>>>(
        min_points.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        orient_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        max_steps_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        unique.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        contour.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        image.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        label_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        real_shape_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        polygons.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    return std::make_tuple(ptr, real_shape_tensor, polygons);
}

__global__ void get_real_polygon_kernel(
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        origin_ptr,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ptr,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>
        real_shape_tensor,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        origin_polygons,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> polygons)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < real_shape_tensor.size(0))
    {
        int origin_start = origin_ptr[idx], origin_end = origin_ptr[idx + 1];
        int start = ptr[idx], end = ptr[idx + 1];

        for (int i = 0; i < real_shape_tensor[idx]; i++)
        {
            polygons[start + i][0] = origin_polygons[origin_start + i][0];
            polygons[start + i][1] = origin_polygons[origin_start + i][1];
        }
        CUDA_ASSERT((end - start) == real_shape_tensor[idx]);
    }
}

std::vector<torch::Tensor> image_to_polygon_cuda(torch::Tensor image)
{
    using namespace torch::indexing;

    CHECK_CUDA(image);
    CHECK_CONTIGUOUS(image);

    auto contour = get_contour(image);

    auto label_tensor = getLabelTensor(image);
    auto unique_tensor = getUniqueValues(label_tensor);
    auto start_points = get_start_points(contour, label_tensor, unique_tensor);

    auto shape_result =
        get_shape_capacity(unique_tensor, contour, label_tensor, start_points);
    auto shape_tensor = std::get<0>(shape_result);
    auto min_points = std::get<1>(shape_result);
    auto orient_tensor = std::get<2>(shape_result);
    auto max_steps_tensor = std::get<3>(shape_result);

    auto trace_result =
        trace_contour(shape_tensor, min_points, orient_tensor, max_steps_tensor,
                      unique_tensor, contour, image, label_tensor);
    auto origin_ptr = std::get<0>(trace_result);
    auto real_shape_tensor = std::get<1>(trace_result);
    auto origin_polygons = std::get<2>(trace_result);

    int total_num = real_shape_tensor.sum().item().toInt();
    torch::Tensor polygons = torch::zeros(
        {total_num, 2}, torch::dtype(torch::kInt).device(image.device()));

    torch::Tensor all_ptr =
        torch::zeros({1 + real_shape_tensor.size(0)},
                     torch::dtype(torch::kInt).device(image.device()));
    all_ptr.index({Slice(1, None)}) += real_shape_tensor;
    all_ptr = all_ptr.cumsum(0).to(torch::kInt);

    cudaSetDevice(label_tensor.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int n = unique_tensor.size(0);
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;

    get_real_polygon_kernel<<<blocks, threads, 0, stream>>>(
        origin_ptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        all_ptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        real_shape_tensor.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        origin_polygons.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        polygons.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

    auto compact_shape_tensor =
        real_shape_tensor.index({real_shape_tensor > 0}).to(torch::kInt);
    torch::Tensor ptr =
        torch::zeros({1 + compact_shape_tensor.size(0)},
                     torch::dtype(torch::kInt).device(image.device()));
    ptr.index({Slice(1, None)}) += compact_shape_tensor;
    ptr = ptr.cumsum(0).to(torch::kInt);

    return {ptr, polygons};
}

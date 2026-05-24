#include <check.h>
#include <polygon.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

std::vector<torch::Tensor> image_to_polygon_cpu(torch::Tensor image)
{
    CHECK_CPU(image);
    CHECK_CONTIGUOUS(image);
    auto image_accessor = image.accessor<float, 2>();
    const int height = image.size(0);
    const int width = image.size(1);
    const float threshold = 0.5;

    auto pack = [](int x, int y) -> long long
    {
        return (static_cast<long long>(x) << 32) | static_cast<unsigned int>(y);
    };
    auto unpack = [](long long key) -> PointCPU<int>
    {
        return PointCPU<int>(static_cast<int>(key >> 32),
                             static_cast<int>(key & 0xffffffff));
    };
    auto is_foreground = [&](int x, int y) -> bool
    {
        return x >= 0 && x < height && y >= 0 && y < width &&
               image_accessor[x][y] > threshold;
    };
    auto add_edge =
        [&](std::unordered_map<long long, std::vector<long long>>& edges,
            int x0, int y0, int x1, int y1)
    { edges[pack(x0, y0)].push_back(pack(x1, y1)); };

    std::unordered_map<long long, std::vector<long long>> edges;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (!is_foreground(i, j))
            {
                continue;
            }
            if (!is_foreground(i - 1, j))
            {
                add_edge(edges, i, j, i, j + 1);
            }
            if (!is_foreground(i, j + 1))
            {
                add_edge(edges, i, j + 1, i + 1, j + 1);
            }
            if (!is_foreground(i + 1, j))
            {
                add_edge(edges, i + 1, j + 1, i + 1, j);
            }
            if (!is_foreground(i, j - 1))
            {
                add_edge(edges, i + 1, j, i, j);
            }
        }
    }

    std::vector<polygon_cpu<int>> polygons;
    for (auto& item : edges)
    {
        while (!item.second.empty())
        {
            long long start = item.first;
            long long cur = item.second.back();
            item.second.pop_back();

            std::vector<PointCPU<int>> raw_points;
            raw_points.push_back(unpack(start));
            int guard = 0;
            while (cur != start && guard++ < (height + 1) * (width + 1) * 4)
            {
                raw_points.push_back(unpack(cur));
                auto it = edges.find(cur);
                if (it == edges.end() || it->second.empty())
                {
                    break;
                }
                cur = it->second.back();
                it->second.pop_back();
            }

            if (cur != start || raw_points.size() < 4)
            {
                continue;
            }

            polygon_cpu<int> polygon;
            const int n = raw_points.size();
            for (int i = 0; i < n; ++i)
            {
                const auto& prev = raw_points[(i + n - 1) % n];
                const auto& point = raw_points[i];
                const auto& next = raw_points[(i + 1) % n];
                const bool vertical = prev.x == point.x && point.x == next.x;
                const bool horizontal = prev.y == point.y && point.y == next.y;
                if (!vertical && !horizontal)
                {
                    polygon.addPoint(point);
                }
            }
            if (polygon.getNumPoints() >= 4)
            {
                polygons.push_back(polygon);
            }
        }
    }

    auto ptr = torch::zeros({(int)polygons.size() + 1}, torch::kInt);
    int total_points = 0;
    for (auto& p : polygons)
    {
        total_points += p.getNumPoints();
    }
    auto tensor = torch::zeros({total_points, 2}, torch::kInt);

    auto ptr_accessor = ptr.accessor<int, 1>();
    auto tensor_accessor = tensor.accessor<int, 2>();

    for (int i = 0; i < polygons.size(); ++i)
    {
        int start = ptr_accessor[i];
        ptr_accessor[i + 1] = polygons[i].getNumPoints() + start;
        for (int j = 0; j < polygons[i].getNumPoints(); ++j)
        {
            tensor_accessor[start + j][0] = polygons[i][j].x;
            tensor_accessor[start + j][1] = polygons[i][j].y;
        }
    }

    return {ptr, tensor};
}

std::vector<torch::Tensor> image_to_polygon_cuda(torch::Tensor image);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("image_to_polygon_cpu", &image_to_polygon_cpu,
          "Convert image to ptr and points tensor");
    m.def("image_to_polygon_cuda", &image_to_polygon_cuda, py::arg("image"));
}

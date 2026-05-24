import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_MODULE = None


def get_image_to_polygon_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    root = Path(__file__).resolve().parents[1]
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    extension_name = "gcontour_image_to_polygon_v2"
    build_dir = root / "build" / "torch_extensions" / extension_name
    build_dir.mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=extension_name,
        sources=[
            str(root / "cpp_extension" / "gcontour.cpp"),
            str(root / "cpp_extension" / "gcontour_kernel.cu"),
        ],
        extra_include_paths=[str(root / "cpp_extension")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        build_directory=str(build_dir),
        verbose=False,
    )
    return _MODULE

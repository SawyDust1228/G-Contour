# G-Contour

G-Contour is a GPU-accelerated contour tracing framework for converting large
binary layout images into polygon contours. This repository contains the Python
evaluation script, benchmark images, and a PyTorch C++/CUDA extension for
single-GPU contour tracing.

## Usage

```bash
python -u eval.py --GPU --device 0 --benchmark gcd_ng_polygon
```

To generate GDS outputs for all benchmarks into `temp/`:

```bash
./run_all.sh
```

`run_all.sh` accepts one optional mode flag:

```bash
./run_all.sh CPU   # run CPU benchmarks only
./run_all.sh GPU   # run GPU benchmarks only
./run_all.sh ALL   # run both CPU and GPU benchmarks
```

If no flag is provided, the script defaults to `ALL`.

## Citation

If you use this work for academic research, please cite:

```bibtex
@inproceedings{g-contour,
  title={G-Contour: GPU Accelerated Contour Tracing For Large-Scale Layouts},
  author={Yin, Shuo and Xu, Jiahao and Jiang, Jiaxi and Li, Mingjun and Ma, Yuzhe and Ho, Tsung-Yi and Yu, Bei},
  booktitle={2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD)},
  pages={1--9},
  year={2025},
  organization={IEEE}
}
```

Note: the open-source implementation does not depend on the OpenCV package.

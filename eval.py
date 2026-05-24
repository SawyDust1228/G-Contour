import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "src")
)
import argparse

from src.contour import Tracer
from src.utils import getAllBenchMarks
from src.utils import buildTempFolder
from src.gds import GDSWriter

# Set GPU as an arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--GPU", action="store_true", default=False)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--no_write", action="store_true", default=False)
parser.add_argument("--benchmark", type=str, default=None)
args = parser.parse_args()
GPU = args.GPU
device_id = args.device if GPU else None
no_write = args.no_write
benchmark_filter = args.benchmark


if __name__ == "__main__":
    benchmarks = getAllBenchMarks()
    if benchmark_filter is not None:
        benchmarks = [name for name in benchmarks if name == benchmark_filter]
        assert len(benchmarks) == 1, f"Benchmark {benchmark_filter} not found"
    tmp_path = buildTempFolder()
    for i in range(len(benchmarks)):
        tracer = Tracer(benchmarks[i], device_id)
        image = tracer.getImage()

        ptr, polygons, runtime  = tracer.trace(image=image, GPU=GPU)
        if not no_write:
            writer = GDSWriter(benchmarks[i], tmp_path, GPU=GPU)
            writer.write(ptr, polygons)
        if GPU:
            print(f"[GPU RUNTIME]: benchmark {benchmarks[i]} using {runtime}s")
        else:
            print(f"[CPU RUNTIME]: benchmark {benchmarks[i]} using {runtime}s")
            

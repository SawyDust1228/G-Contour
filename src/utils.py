from pathlib import Path
import os

def getBenchmarkPath():
    return str(Path(__file__).resolve().parents[1] / "benchmark")

def getAllBenchMarks():
    benchmark_path = Path(getBenchmarkPath())
    names = []
    for file_path in benchmark_path.glob('*.png'):
        names.append(file_path.stem)
    return names

def getFolderPath():
    return str(Path(__file__).resolve().parents[1])

def buildTempFolder():
    path = getFolderPath() + "/temp"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

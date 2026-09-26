# Param: RXMesh mesh parameterization

with gradient descent. 

## Requirements

- CMake 3.25 or newer
- A C++17 compiler and CUDA toolkit
- An NVIDIA GPU to run the application

RXMesh and its dependencies are fetched during CMake configuration. The input must be a triangular OBJ mesh with a boundary.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target Param --parallel
```

The default CUDA architecture is `native`. For a headless builder or cross compilation, specify a numeric architecture, for example `-DCMAKE_CUDA_ARCHITECTURES=89`.

Polyscope visualization is enabled by default. To build without it, configure with `-DRX_USE_POLYSCOPE=OFF`.

## Run

On Linux:

```bash
./build/bin/Param --input path/to/open_mesh.obj --lr 1e-9 --iter 100
```
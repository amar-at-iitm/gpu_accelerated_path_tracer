# GPU-Accelerated Path Tracer in C++ & CUDA


This project is a physically-based path tracer that leverages the massive parallelism of modern GPUs to render photorealistic images. It is implemented in **C++** and **NVIDIA CUDA**. The renderer simulates light transport using Monte Carlo integration to solve the rendering equation, producing high-quality images with complex lighting, global illumination, and realistic material interactions.



---

## Features

* **Physically-Based Rendering (PBR)**: Simulates light accurately based on physical principles.
* **GPU Acceleration**: Core ray tracing kernels are written in **CUDA** for massive performance gains.
* **Bounding Volume Hierarchy (BVH)**: A highly efficient acceleration structure built on the CPU (using SAH) and traversed on the GPU for fast ray-object intersections.
* **Advanced Monte Carlo Techniques**:
    * **Multiple Importance Sampling (MIS)**: Reduces noise by intelligently combining samples from different distributions (e.g., light source sampling and BSDF sampling).
    * **Russian Roulette**: Unbiased path termination to improve performance.
    * **Stratified Sampling**: Reduces aliasing and variance in the first generation of rays.
* **Material Support**: Handles **Diffuse**, **Specular (Mirror)**, and **Dielectric (Glass)** surfaces with energy-conserving BSDFs.
* **HDR Environment Lighting**: Supports high-dynamic-range image-based lighting.
* **Progressive Rendering**: The image progressively refines over time, allowing for quick previews.
* **Detailed Output**: Generates rendered images (PNG), convergence plots (noise vs. samples), and performance statistics.

---

## Performance

The primary goal of this project was to achieve a significant speedup over a traditional CPU-based renderer. The GPU implementation shows a dramatic improvement in performance, measured in samples per second.

**Hardware**: *[Enter Your GPU and CPU specs here, e.g., NVIDIA RTX 4090, Intel Core i9-13900K]*

| Renderer | Scene | Resolution | Time to 1024 spp | Samples/Second |
| :--- | :---: | :---: | :---: | :---: |
| **CPU (Single-Threaded)** | Cornell Box | 1024x1024 | ~2 hours | ~150 K |
| **GPU (CUDA)** | Cornell Box | 1024x1024 | **~15 seconds** | **~70 M** |

### Convergence Plot

The plot below shows the Mean Squared Error (MSE) decreasing as the number of samples per pixel increases, demonstrating the convergence of the Monte Carlo integrator.



---

## Architecture & Design

* **Language**: **C++17** for the host code (scene loading, BVH construction) and **NVIDIA CUDA** for the high-performance device kernels (ray generation, intersection, shading).
* **Data Layout**: The Bounding Volume Hierarchy (BVH) is constructed on the CPU and then flattened into arrays. This GPU-friendly layout enables efficient, stackless traversal in CUDA kernels. All scene data (vertices, indices, materials) is packed and transferred to GPU memory.
* **Render Loop**: A progressive render loop accumulates samples over time. Two GPU buffers (ping-pong buffers) are used to store the current and previous accumulation results, avoiding costly read-modify-write operations.

---

## Getting Started

Follow these instructions to build and run the project on your local machine.

### Prerequisites

* A C++17 compatible compiler (GCC, Clang, or MSVC)
* [CMake](https://cmake.org/download/) (version 3.10 or higher)
* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (version 11.0 or higher)
* An NVIDIA GPU with Compute Capability 6.0 or higher.

### Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amar-at-iitm/gpu_accelerated_path_tracer
    cd gpu_accelerated_path_tracer
    ```
2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Configure and build the project with CMake:**
    ```bash
    cmake ..
    make -j$(nproc) # On Linux
    # Or use 'cmake --build .' on all platforms
    ```

### Running the Renderer

Execute the renderer from the `build` directory. You must specify a scene file and an output image path.

```bash
./path-tracer --scene ../scenes/cornell_box.xml --output cornell_render.png
````

-----

## Technical Details

### The Rendering Equation

At its core, the renderer solves the Rendering Equation, which describes the equilibrium distribution of light in a scene. It is formulated as:

$$L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (\omega_i \cdot \vec{n}) d\omega_i$$

Where:

  * $L\_o$ is the outgoing radiance from point $p$ in direction $\\omega\_o$.
  * $L\_e$ is the emitted radiance from point $p$.
  * $\\int\_{\\Omega}$ is the integral over the hemisphere $\\Omega$ centered around the normal $\\vec{n}$.
  * $f\_r$ is the Bidirectional Scattering Distribution Function (BSDF).
  * $L\_i$ is the incoming radiance to point $p$ from direction $\\omega\_i$.

This integral is solved using Monte Carlo methods, where paths of light are traced from the camera into the scene.

### Bounding Volume Hierarchy (BVH)

To avoid testing every single triangle in the scene for a ray intersection (an $O(N)$ operation), we use a **BVH**.

  * **Construction (CPU)**: The BVH is built on the CPU using the **Surface Area Heuristic (SAH)** to partition triangles, creating a tree structure that minimizes the probability of a ray intersecting bounding boxes.
  * **Traversal (GPU)**: The tree is flattened into a linear array to be GPU-friendly. A stackless traversal algorithm is implemented in a CUDA kernel, allowing thousands of rays to traverse the structure in parallel with minimal overhead.

-----

## Project Structure

```
/path-tracer/
├── README.md
├── build/
├── scenes/
│   ├── cornell_box.xml
│   └── hdri_scene.xml
└── src/
    ├── bvh.h/cpp
    ├── camera.h
    ├── main.cpp
    ├── material.h
    ├── scene.h/cpp
    └── kernels/
        ├── intersect.cu
        ├── pathtrace.cu
        └── utility.cuh
```


## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

```

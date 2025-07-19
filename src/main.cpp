// src/main.cpp 


#include <iostream>
#include <vector>
#include "scene.h"
#include "bvh.h"
#include "cuda_runtime.h"

// Forward declaration of the CUDA kernel
extern "C" void path_trace_kernel(Vec3* image_buffer, int width, int height, int samples_per_pixel, Camera cam, BVHNode* bvh_nodes, Triangle* triangles, int* prim_indices, Material* materials);

// Function to save the rendered image buffer to a PPM file
void save_image(const std::string& filepath, int width, int height, Vec3* buffer) {
    std::ofstream out(filepath);
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        // Apply gamma correction
        float r = sqrtf(buffer[i].x);
        float g = sqrtf(buffer[i].y);
        float b = sqrtf(buffer[i].z);
        out << (int)(255.99 * r) << " " << (int)(255.99 * g) << " " << (int)(255.99 * b) << "\n";
    }
}

int main(int argc, char** argv) {
    const int WIDTH = 800;
    const int HEIGHT = 800;
    const int SAMPLES_PER_PIXEL = 128;

    // 1. Load Scene
    Scene scene;
    if (!scene.load("scenes/cornell_box.xml")) {
        return -1;
    }

    // 2. Build BVH
    BVHBuilder bvh;
    bvh.build(scene.triangles);

    // 3. Allocate GPU Memory
    Vec3* d_image_buffer;
    BVHNode* d_bvh_nodes;
    Triangle* d_triangles;
    int* d_prim_indices;
    Material* d_materials;

    cudaMalloc(&d_image_buffer, WIDTH * HEIGHT * sizeof(Vec3));
    cudaMalloc(&d_bvh_nodes, bvh.get_flat_nodes().size() * sizeof(BVHNode));
    cudaMalloc(&d_triangles, scene.triangles.size() * sizeof(Triangle));
    cudaMalloc(&d_prim_indices, bvh.get_prim_indices().size() * sizeof(int));
    cudaMalloc(&d_materials, scene.materials.size() * sizeof(Material));

    // 4. Copy Data to GPU
    cudaMemcpy(d_bvh_nodes, bvh.get_flat_nodes().data(), bvh.get_flat_nodes().size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, scene.triangles.data(), scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prim_indices, bvh.get_prim_indices().data(), bvh.get_prim_indices().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    // 5. Launch Kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    std::cout << "Rendering..." << std::endl;
    path_trace_kernel<<<numBlocks, threadsPerBlock>>>(d_image_buffer, WIDTH, HEIGHT, SAMPLES_PER_PIXEL, scene.camera, d_bvh_nodes, d_triangles, d_prim_indices, d_materials);
    cudaDeviceSynchronize();
    std::cout << "Rendering complete." << std::endl;

    // 6. Copy Result Back to CPU and Save
    std::vector<Vec3> host_image_buffer(WIDTH * HEIGHT);
    cudaMemcpy(host_image_buffer.data(), d_image_buffer, WIDTH * HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost);
    save_image("render.ppm", WIDTH, HEIGHT, host_image_buffer.data());
    std::cout << "Image saved to render.ppm" << std::endl;

    // 7. Free GPU Memory
    cudaFree(d_image_buffer);
    cudaFree(d_bvh_nodes);
    cudaFree(d_triangles);
    cudaFree(d_prim_indices);
    cudaFree(d_materials);

    return 0;
}

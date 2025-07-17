// src/kernels/utility.cuh


#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// A small constant for floating-point comparisons to avoid precision errors
#define EPSILON 1e-5f
#define PI 3.1415926535f

// --- Core Data Structures ---

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return {x * v.x, y * v.y, z * v.z}; }
    __host__ __device__ Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ float squared_length() const { return x * x + y * y + z * z; }
};

__host__ __device__ inline Vec3 normalize(const Vec3& v) { return v / v.length(); }
__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct HitRecord {
    float t = FLT_MAX; // Distance to intersection
    Vec3 p;            // Intersection point
    Vec3 normal;       // Surface normal at intersection
    int material_id;   // ID of the material hit
};

struct AABB { // Axis-Aligned Bounding Box
    Vec3 min_bounds;
    Vec3 max_bounds;
};

// --- GPU-Friendly Scene Objects ---

struct Triangle {
    Vec3 v0, v1, v2;
    int material_id;
};

// Flattened BVH node for GPU traversal.
// Using indices instead of pointers is crucial for GPU data structures.
struct BVHNode {
    AABB bounds;
    int prim_offset; // If leaf, offset into the primitive indices array. If internal, offset to left child.
    int prim_count;  // If > 0, this is a leaf node. If 0, it's an internal node. Right child is at prim_offset + 1.
};

// --- Random Number Helper ---

// Generates a random direction within a unit sphere for diffuse scattering
__device__ inline Vec3 random_in_unit_sphere(curandState* local_rand_state) {
    Vec3 p;
    do {
        p = {curand_uniform(local_rand_state) * 2.0f - 1.0f,
             curand_uniform(local_rand_state) * 2.0f - 1.0f,
             curand_uniform(local_rand_state) * 2.0f - 1.0f};
    } while (p.squared_length() >= 1.0f);
    return p;
}
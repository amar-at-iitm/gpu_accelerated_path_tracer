//  src/material.h


#pragma once

#include "kernels/utility.cuh"

// Enum to identify material types in our CUDA kernels
enum MaterialType {
    DIFFUSE,
    SPECULAR, // Perfect mirror
    DIELECTRIC  // Glass, water, etc.
};

// The material struct that will be passed to the GPU.
// It holds properties common to all materials.
struct Material {
    MaterialType type;
    Vec3 albedo;       // Color for diffuse and specular
    float ref_idx;     // Refractive index for dielectric
};

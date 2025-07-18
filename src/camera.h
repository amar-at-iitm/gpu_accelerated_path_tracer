// src/camera.h


#pragma once

#include "kernels/utility.cuh"

// Defines the camera's position and orientation in the scene.
struct Camera {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;

    // Default constructor
    Camera() {}

    // Constructor to set up the camera based on view parameters
    Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect) {
        origin = lookfrom;
        float theta = vfov * PI / 180.0f;
        float half_height = tanf(theta / 2.0f);
        float half_width = aspect * half_height;

        Vec3 w = normalize(lookfrom - lookat);
        Vec3 u = normalize(cross(vup, w));
        Vec3 v = cross(w, u);

        lower_left_corner = origin - u * half_width - v * half_height - w;
        horizontal = u * 2.0f * half_width;
        vertical = v * 2.0f * half_height;
    }
};

// This device function calculates the ray for a given pixel coordinate (u, v)
__device__ inline Ray get_camera_ray(const Camera& cam, float u, float v) {
    return {cam.origin, normalize(cam.lower_left_corner + cam.horizontal * u + cam.vertical * v - cam.origin)};
}

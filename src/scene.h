// src/scene.h

#pragma once

#include <vector>
#include <string>
#include "kernels/utility.cuh"
#include "camera.h"
#include "material.h"

// The Scene class orchestrates loading and storing all scene data.
class Scene {
public:
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    Camera camera;

    Scene() = default;

    // Main function to load a scene from an XML file
    bool load(const std::string& filepath);
};

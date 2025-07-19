//  src/scene.cpp

#include "scene.h"
#include <fstream>
#include <iostream>
#include <sstream>

// Basic string splitting helper
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Naive XML parsing - not robust!
bool Scene::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open scene file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;

        if (tag.find("<camera") != std::string::npos) {
            // Example: <camera lookfrom="0 0 5" lookat="0 0 0" vup="0 1 0" vfov="40" aspect="1.0" />
            // In a real parser, you'd extract these attributes properly.
            Vec3 lookfrom = {0, 1, 6};
            Vec3 lookat = {0, 1, 0};
            Vec3 vup = {0, 1, 0};
            camera = Camera(lookfrom, lookat, vup, 40.0f, 1.0f);
        } else if (tag.find("<mesh") != std::string::npos) {
            // Example: <mesh file="cornell_box.obj" material_id="0" />
            // TODO: Add mesh loading logic (e.g., using tinyobjloader)
            // For now, we manually add triangles for a Cornell Box.

            // Floor
            triangles.push_back({{ -1, 0, -1 }, { 1, 0, -1 }, { 1, 0, 1 }, 0});
            triangles.push_back({{ -1, 0, -1 }, { 1, 0, 1 }, { -1, 0, 1 }, 0});
            // etc... for all other triangles in the scene
        } else if (tag.find("<material") != std::string::npos) {
            // Example: <material type="diffuse" albedo="0.8 0.8 0.8" />
            materials.push_back({DIFFUSE, {0.8f, 0.8f, 0.8f}, 0.0f});
        }
    }
    
    std::cout << "Scene loaded with " << triangles.size() << " triangles and " << materials.size() << " materials." << std::endl;
    return true;
}

// src/bvh.h


#pragma once

#include <vector>
#include <memory>
#include "kernels/utility.cuh"

// This is the CPU-side representation of a BVH node used during construction.
struct BVHBuildNode {
    AABB bounds;
    std::unique_ptr<BVHBuildNode> left = nullptr;
    std::unique_ptr<BVHBuildNode> right = nullptr;
    int first_prim;
    int prim_count;
};

// The BVHBuilder class handles the construction of the BVH on the CPU.
class BVHBuilder {
public:
    BVHBuilder() = default;

    // Build the BVH from a list of triangles
    void build(const std::vector<Triangle>& tris);

    // Get the flattened BVH nodes ready for GPU upload
    const std::vector<BVHNode>& get_flat_nodes() const { return flat_nodes; }
    const std::vector<int>& get_prim_indices() const { return prim_indices; }

private:
    std::unique_ptr<BVHBuildNode> root = nullptr;
    std::vector<Triangle> triangles;
    std::vector<int> prim_indices;
    std::vector<BVHNode> flat_nodes;

    // Recursive build function
    std::unique_ptr<BVHBuildNode> recursive_build(int start, int end);

    // Flattens the tree structure into a GPU-friendly linear array
    int flatten(const std::unique_ptr<BVHBuildNode>& node);
};

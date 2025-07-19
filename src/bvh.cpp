// src/bvh.cpp

#include "bvh.h"
#include <algorithm>
#include <stack>

// Helper function to compute the bounding box of a single triangle
AABB get_triangle_bounds(const Triangle& tri) {
    Vec3 min_v = {
        fminf(tri.v0.x, fminf(tri.v1.x, tri.v2.x)),
        fminf(tri.v0.y, fminf(tri.v1.y, tri.v2.y)),
        fminf(tri.v0.z, fminf(tri.v1.z, tri.v2.z))
    };
    Vec3 max_v = {
        fmaxf(tri.v0.x, fmaxf(tri.v1.x, tri.v2.x)),
        fmaxf(tri.v0.y, fmaxf(tri.v1.y, tri.v2.y)),
        fmaxf(tri.v0.z, fmaxf(tri.v1.z, tri.v2.z))
    };
    return {min_v, max_v};
}

// Helper function to merge two bounding boxes
AABB merge_bounds(const AABB& b1, const AABB& b2) {
    return {
        {fminf(b1.min_bounds.x, b2.min_bounds.x), fminf(b1.min_bounds.y, b2.min_bounds.y), fminf(b1.min_bounds.z, b2.min_bounds.z)},
        {fmaxf(b1.max_bounds.x, b2.max_bounds.x), fmaxf(b1.max_bounds.y, b2.max_bounds.y), fmaxf(b1.max_bounds.z, b2.max_bounds.z)}
    };
}

void BVHBuilder::build(const std::vector<Triangle>& tris) {
    triangles = tris;
    prim_indices.resize(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        prim_indices[i] = i;
    }

    root = recursive_build(0, triangles.size());
    
    // After building the tree, flatten it for the GPU
    flat_nodes.reserve(triangles.size() * 2);
    flatten(root);
}

std::unique_ptr<BVHBuildNode> BVHBuilder::recursive_build(int start, int end) {
    auto node = std::make_unique<BVHBuildNode>();
    int prim_count = end - start;

    // Calculate the bounding box for all primitives in this range
    AABB total_bounds = get_triangle_bounds(triangles[prim_indices[start]]);
    for (int i = start + 1; i < end; ++i) {
        total_bounds = merge_bounds(total_bounds, get_triangle_bounds(triangles[prim_indices[i]]));
    }
    node->bounds = total_bounds;
    node->prim_count = prim_count;
    node->first_prim = start;

    if (prim_count <= 4) { // Leaf node condition
        return node;
    }

    // Find the best axis and split point using SAH (Simplified version)
    int best_axis = -1;
    int best_split = -1;
    float best_cost = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis) {
        // Sort primitives along the current axis
        std::sort(prim_indices.begin() + start, prim_indices.begin() + end, [&](int a, int b) {
            Vec3 center_a = (triangles[a].v0 + triangles[a].v1 + triangles[a].v2) / 3.0f;
            Vec3 center_b = (triangles[b].v0 + triangles[b].v1 + triangles[b].v2) / 3.0f;
            return ((float*)&center_a)[axis] < ((float*)&center_b)[axis];
        });

        // Test splits
        for (int i = start + 1; i < end; ++i) {
            // Simplified cost: just split in the middle
            float cost = (float)i + (float)(end - i); // Placeholder for real SAH cost
            if (cost < best_cost) {
                best_cost = cost;
                best_split = i;
                best_axis = axis;
            }
        }
    }

    // Re-sort along the best axis before splitting
     std::sort(prim_indices.begin() + start, prim_indices.begin() + end, [&](int a, int b) {
            Vec3 center_a = (triangles[a].v0 + triangles[a].v1 + triangles[a].v2) / 3.0f;
            Vec3 center_b = (triangles[b].v0 + triangles[b].v1 + triangles[b].v2) / 3.0f;
            return ((float*)&center_a)[best_axis] < ((float*)&center_b)[best_axis];
        });


    node->left = recursive_build(start, best_split);
    node->right = recursive_build(best_split, end);
    node->prim_count = 0; // Mark as internal node

    return node;
}

int BVHBuilder::flatten(const std::unique_ptr<BVHBuildNode>& node) {
    int current_idx = flat_nodes.size();
    flat_nodes.emplace_back(); // Reserve space

    flat_nodes[current_idx].bounds = node->bounds;
    flat_nodes[current_idx].prim_count = node->prim_count;
    
    if (node->prim_count > 0) { // Leaf node
        flat_nodes[current_idx].prim_offset = node->first_prim;
    } else { // Internal node
        flatten(node->left);
        flat_nodes[current_idx].prim_offset = flatten(node->right) - 1;
    }
    
    return current_idx;
}

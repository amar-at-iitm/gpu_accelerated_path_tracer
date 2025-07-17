// src/kernels/intersect.cu

#include "utility.cuh"

// --- Ray-Geometry Intersection Kernels ---

// Fast Ray-AABB (slab test) intersection
__device__ bool intersect_aabb(const Ray& r, const Vec3& inv_dir, const AABB& box) {
    float tmin = (box.min_bounds.x - r.origin.x) * inv_dir.x;
    float tmax = (box.max_bounds.x - r.origin.x) * inv_dir.x;
    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (box.min_bounds.y - r.origin.y) * inv_dir.y;
    float tymax = (box.max_bounds.y - r.origin.y) * inv_dir.y;
    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax)) return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = (box.min_bounds.z - r.origin.z) * inv_dir.z;
    float tzmax = (box.max_bounds.z - r.origin.z) * inv_dir.z;
    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax)) return false;

    return true;
}

// Ray-Triangle intersection using the Möller-Trumbore algorithm
__device__ bool intersect_triangle(const Ray& r, const Triangle& tri, HitRecord& rec) {
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;
    Vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return false; // Ray is parallel to the triangle

    float f = 1.0f / a;
    Vec3 s = r.origin - tri.v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    // We have an intersection, now compute t
    float t = f * dot(edge2, q);
    if (t > EPSILON && t < rec.t) { // Check if this hit is closer than the previous one
        rec.t = t;
        rec.p = r.origin + r.direction * t;
        rec.normal = normalize(cross(edge1, edge2));
        rec.material_id = tri.material_id;
        return true;
    }
    return false;
}


// --- Main BVH Traversal Function ---

// This is the 'trace' function called from the main path tracing kernel.
// It performs an iterative (stackless) traversal of the BVH.
__device__ bool trace(const Ray& r, HitRecord& rec, const BVHNode* bvh_nodes, const Triangle* triangles, const int* prim_indices) {
    bool hit_anything = false;
    Vec3 inv_dir = {1.0f / r.direction.x, 1.0f / r.direction.y, 1.0f / r.direction.z};
    
    // Traversal stack stored in a small, fixed-size array.
    // For most scenes, a stack size of 32-64 is more than enough.
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Start traversal at the root node (index 0)

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = bvh_nodes[node_idx];

        // Don't bother checking nodes that are further away than our current closest hit
        if (!intersect_aabb(r, inv_dir, node.bounds)) continue;

        if (node.prim_count > 0) { // This is a LEAF node
            for (int i = 0; i < node.prim_count; ++i) {
                int tri_idx = prim_indices[node.prim_offset + i];
                if (intersect_triangle(r, triangles[tri_idx], rec)) {
                    hit_anything = true;
                }
            }
        } else { // This is an INTERNAL node
            // Push children onto the traversal stack.
            // A small optimization could be to check the closer child first.
            stack[stack_ptr++] = node.prim_offset;      // Left child
            stack[stack_ptr++] = node.prim_offset + 1;  // Right child
        }
    }

    return hit_anything;
}
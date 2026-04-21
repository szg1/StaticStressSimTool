#ifndef MESH_H
#define MESH_H

#include <vector>
#include <string>
#include "geometry.h"

struct Mesh {
    std::vector<Triangle> triangles;
    BoundingBox bounds;
    Vec3 center;
    Vec3 color;
    float alpha = 1.0f; // Transparency
    Vec3 positionOffset; // World space offset
    Vec3 rotation = {0, 0, 0}; // Euler angles in degrees
    bool active = false;

    // Hole detection
    bool showHole = false;
    struct HoleLine {
        Vec3 top;
        Vec3 bottom;
    };
    std::vector<HoleLine> holes;

    // Slicing support
    std::vector<Vec3> vertexNormals; // Per-vertex averaged normals
    std::vector<Vec3> vertexColors;  // Optional: Per-vertex colors

    void updateBounds(const Vec3& v);
    bool loadFromSTL(const char* filename);
    void draw(bool useInternalColor = true, bool applyTransform = true) const;
    void drawBoundingBox() const;

    // Painting support
    void initVertexColors(const Vec3& c);
    void paintVertices(const Vec3& modelHitPoint, float radius, const Vec3& paintColor);
    bool intersectRay(const Vec3& rayOrigin, const Vec3& rayDir, Vec3& hitPoint, float& tMin) const;

    // Attempt to detect a vertical hole and set holeTop/holeBottom
    void detectHole();

    // Calculate the Z coordinate (local) where the screw head ends and shaft begins
    float getHeadBottomZ() const;

    // Calculate the Z/Y offset required to drop the screw into the hole until contact
    float computeDropDistance(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const;

    // Iterative binary-search-like placement
    Vec3 findGravitySeating(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const;
    Vec3 getAlignedPosition(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const;
    bool checkCollision(const Mesh& other) const;

    // Slicing methods
    void computeVertexNormals();
    Mesh generateHollow(float wallThickness, float bottomThick, float topThick) const;

    // New methods for Infill
    Mesh generateInnerShell(float wallThickness, float bottomThick, float topThick) const;
    Mesh generateInfill(float nozzleWidth, float layerHeight, float infillPercent) const;

    void refineMesh(float maxArea);

    // Merging support
    void addMesh(const Mesh& other);
};

#endif

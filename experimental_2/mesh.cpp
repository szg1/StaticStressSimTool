#include "mesh.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void Mesh::updateBounds(const Vec3& v) {
    if (v.x < bounds.minX) bounds.minX = v.x;
    if (v.x > bounds.maxX) bounds.maxX = v.x;
    if (v.y < bounds.minY) bounds.minY = v.y;
    if (v.y > bounds.maxY) bounds.maxY = v.y;
    if (v.z < bounds.minZ) bounds.minZ = v.z;
    if (v.z > bounds.maxZ) bounds.maxZ = v.z;
}

bool Mesh::loadFromSTL(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    triangles.clear();
    bounds.reset();
    showHole = false;

    char header[80];
    file.read(header, 80);

    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(numTriangles));

    std::cout << "Reading " << numTriangles << " triangles from " << filename << "..." << std::endl;

    for (uint32_t i = 0; i < numTriangles; ++i) {
        Triangle t;
        file.read(reinterpret_cast<char*>(&t.normal), sizeof(Vec3));
        file.read(reinterpret_cast<char*>(&t.v1), sizeof(Vec3));
        file.read(reinterpret_cast<char*>(&t.v2), sizeof(Vec3));
        file.read(reinterpret_cast<char*>(&t.v3), sizeof(Vec3));

        unsigned short attribute;
        file.read(reinterpret_cast<char*>(&attribute), sizeof(unsigned short));

        updateBounds(t.v1);
        updateBounds(t.v2);
        updateBounds(t.v3);

        triangles.push_back(t);
    }

    center.x = bounds.minX + bounds.width() / 2.0f;
    center.y = bounds.minY + bounds.height() / 2.0f;
    center.z = bounds.minZ + bounds.depth() / 2.0f;

    active = true;
    return true;
}

void Mesh::draw(bool useInternalColor, bool applyTransform) const {
    if (!active) return;

    glPushMatrix();

    if (applyTransform) {
        glTranslatef(positionOffset.x, positionOffset.y, positionOffset.z);
        glRotatef(rotation.x, 1.0f, 0.0f, 0.0f);
        glRotatef(rotation.y, 0.0f, 1.0f, 0.0f);
        glRotatef(rotation.z, 0.0f, 0.0f, 1.0f);
        glTranslatef(-center.x, -center.y, -center.z);
    }

    if (useInternalColor && vertexColors.empty()) {
        glColor4f(color.x, color.y, color.z, alpha);
    }

    glBegin(GL_TRIANGLES);
    bool useVC = !vertexColors.empty();
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& t = triangles[i];
        glNormal3f(t.normal.x, t.normal.y, t.normal.z);

        if (useVC) glColor4f(vertexColors[3*i].x, vertexColors[3*i].y, vertexColors[3*i].z, alpha);
        glVertex3f(t.v1.x, t.v1.y, t.v1.z);

        if (useVC) glColor4f(vertexColors[3*i+1].x, vertexColors[3*i+1].y, vertexColors[3*i+1].z, alpha);
        glVertex3f(t.v2.x, t.v2.y, t.v2.z);

        if (useVC) glColor4f(vertexColors[3*i+2].x, vertexColors[3*i+2].y, vertexColors[3*i+2].z, alpha);
        glVertex3f(t.v3.x, t.v3.y, t.v3.z);
    }
    glEnd();

    if (showHole) {
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glColor3f(0.0f, 1.0f, 0.0f);
        glLineWidth(3.0f);

        glBegin(GL_LINES);
        for (const auto& hole : holes) {
            glVertex3f(hole.top.x, hole.top.y, hole.top.z);
            glVertex3f(hole.bottom.x, hole.bottom.y, hole.bottom.z);
        }
        glEnd();

        glLineWidth(1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
    }

    glPopMatrix();
}

void Mesh::drawBoundingBox() const {
    if (!active) return;

    glPushMatrix();
    glTranslatef(positionOffset.x, positionOffset.y, positionOffset.z);
    glTranslatef(-center.x, -center.y, -center.z);

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 0.0f);
    glLineWidth(2.0f);

    float x1 = bounds.minX, x2 = bounds.maxX;
    float y1 = bounds.minY, y2 = bounds.maxY;
    float z1 = bounds.minZ, z2 = bounds.maxZ;

    glBegin(GL_LINES);
    glVertex3f(x1, y1, z1); glVertex3f(x2, y1, z1);
    glVertex3f(x2, y1, z1); glVertex3f(x2, y2, z1);
    glVertex3f(x2, y2, z1); glVertex3f(x1, y2, z1);
    glVertex3f(x1, y2, z1); glVertex3f(x1, y1, z1);

    glVertex3f(x1, y1, z2); glVertex3f(x2, y1, z2);
    glVertex3f(x2, y1, z2); glVertex3f(x2, y2, z2);
    glVertex3f(x2, y2, z2); glVertex3f(x1, y2, z2);
    glVertex3f(x1, y2, z2); glVertex3f(x1, y1, z2);

    glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z2);
    glVertex3f(x2, y1, z1); glVertex3f(x2, y1, z2);
    glVertex3f(x2, y2, z1); glVertex3f(x2, y2, z2);
    glVertex3f(x1, y2, z1); glVertex3f(x1, y2, z2);
    glEnd();

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

// --- Helper Structs and Functions ---

namespace {
// Helper for Barycentric coordinates
bool isPointInTriangle3D(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    // Project to 2D plane of the triangle? Or use 3D barycentric?
    // Using barycentric technique:
    // v0 = b - a, v1 = c - a, v2 = p - a
    Vec3 v0 = b - a;
    Vec3 v1 = c - a;
    Vec3 v2 = p - a;

    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);

    float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-6f) return false; // Degenerate

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return (v >= 0.0f) && (w >= 0.0f) && (u >= 0.0f);
}

bool segmentIntersectsTriangle(const Vec3& p1, const Vec3& p2, const Triangle& t) {
    const float kEpsilon = 1e-8f;
    Vec3 rayDir = p2 - p1;
    float len = rayDir.length();
    if (len < kEpsilon) return false;
    rayDir = rayDir / len; // Normalize

    Vec3 v0 = t.v1;
    Vec3 v1 = t.v2;
    Vec3 v2 = t.v3;

    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = rayDir.cross(edge2);
    float a = edge1.dot(h);

    if (a > -kEpsilon && a < kEpsilon) return false; // Parallel

    float f = 1.0f / a;
    Vec3 s = p1 - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = s.cross(edge1);
    float v = f * rayDir.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float tVal = f * edge2.dot(q);

    if (tVal >= -1e-5f && tVal <= len + 1e-5f) return true; // Intersection within segment
    return false;
}

bool checkTriTri(const Triangle& t1, const Triangle& t2) {
    // Check edges of t1 against t2
    if (segmentIntersectsTriangle(t1.v1, t1.v2, t2)) return true;
    if (segmentIntersectsTriangle(t1.v2, t1.v3, t2)) return true;
    if (segmentIntersectsTriangle(t1.v3, t1.v1, t2)) return true;

    // Check edges of t2 against t1
    if (segmentIntersectsTriangle(t2.v1, t2.v2, t1)) return true;
    if (segmentIntersectsTriangle(t2.v2, t2.v3, t1)) return true;
    if (segmentIntersectsTriangle(t2.v3, t2.v1, t1)) return true;

    return false;
}
}

struct Segment {
    Vec3 p1, p2;
};

std::vector<Segment> sliceMeshZ(const std::vector<Triangle>& triangles, float zPlane) {
    std::vector<Segment> segments;
    for (const auto& t : triangles) {
        float d1 = t.v1.z - zPlane;
        float d2 = t.v2.z - zPlane;
        float d3 = t.v3.z - zPlane;

        int pos = 0, neg = 0, zero = 0;
        if (d1 > 0) pos++; else if (d1 < 0) neg++; else zero++;
        if (d2 > 0) pos++; else if (d2 < 0) neg++; else zero++;
        if (d3 > 0) pos++; else if (d3 < 0) neg++; else zero++;

        if (pos > 0 && neg > 0) {
            Vec3 pts[2];
            int count = 0;
            if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
                float f = d1 / (d1 - d2);
                pts[count++] = t.v1 + (t.v2 - t.v1) * f;
            }
            if ((d2 > 0 && d3 < 0) || (d2 < 0 && d3 > 0)) {
                float f = d2 / (d2 - d3);
                pts[count++] = t.v2 + (t.v3 - t.v2) * f;
            }
            if ((d3 > 0 && d1 < 0) || (d3 < 0 && d1 > 0)) {
                float f = d3 / (d3 - d1);
                pts[count++] = t.v3 + (t.v1 - t.v3) * f;
            }

            if (count == 2) {
                segments.push_back({pts[0], pts[1]});
            }
        }
    }
    return segments;
}

std::vector<Segment> sliceMeshY(const std::vector<Triangle>& triangles, float yPlane) {
    std::vector<Segment> segments;
    for (const auto& t : triangles) {
        float d1 = t.v1.y - yPlane;
        float d2 = t.v2.y - yPlane;
        float d3 = t.v3.y - yPlane;

        int pos = 0, neg = 0, zero = 0;
        if (d1 > 0) pos++; else if (d1 < 0) neg++; else zero++;
        if (d2 > 0) pos++; else if (d2 < 0) neg++; else zero++;
        if (d3 > 0) pos++; else if (d3 < 0) neg++; else zero++;

        if (pos > 0 && neg > 0) {
            Vec3 pts[2];
            int count = 0;
            if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
                float f = d1 / (d1 - d2);
                pts[count++] = t.v1 + (t.v2 - t.v1) * f;
            }
            if ((d2 > 0 && d3 < 0) || (d2 < 0 && d3 > 0)) {
                float f = d2 / (d2 - d3);
                pts[count++] = t.v2 + (t.v3 - t.v2) * f;
            }
            if ((d3 > 0 && d1 < 0) || (d3 < 0 && d1 > 0)) {
                float f = d3 / (d3 - d1);
                pts[count++] = t.v3 + (t.v1 - t.v3) * f;
            }

            if (count == 2) {
                segments.push_back({pts[0], pts[1]});
            }
        }
    }
    return segments;
}

std::vector<std::vector<Vec3>> segmentsToLoops(const std::vector<Segment>& segments) {
    std::vector<std::vector<Vec3>> loops;
    if (segments.empty()) return loops;

    const float epsilon = 0.001f;
    std::vector<bool> visited(segments.size(), false);

    for (size_t i = 0; i < segments.size(); ++i) {
        if (visited[i]) continue;

        std::vector<Vec3> loopPoints;
        loopPoints.push_back(segments[i].p1);
        loopPoints.push_back(segments[i].p2);
        visited[i] = true;

        Vec3 currentEnd = segments[i].p2;
        bool foundNext = true;
        while (foundNext) {
            foundNext = false;
            for (size_t j = 0; j < segments.size(); ++j) {
                if (visited[j]) continue;

                if ((segments[j].p1 - currentEnd).length() < epsilon) {
                    currentEnd = segments[j].p2;
                    loopPoints.push_back(currentEnd);
                    visited[j] = true;
                    foundNext = true;
                    break;
                } else if ((segments[j].p2 - currentEnd).length() < epsilon) {
                    currentEnd = segments[j].p1;
                    loopPoints.push_back(currentEnd);
                    visited[j] = true;
                    foundNext = true;
                    break;
                }
            }
        }
        loops.push_back(loopPoints);
    }
    return loops;
}

bool isPointInPolygon(const Vec3& p, const std::vector<std::vector<Vec3>>& loops) {
    int intersections = 0;
    // Ray cast to +X
    for (const auto& loop : loops) {
        if (loop.size() < 2) continue;
        for (size_t i = 0; i < loop.size() - 1; ++i) {
            Vec3 v1 = loop[i];
            Vec3 v2 = loop[i+1];

            // Check if edge intersects ray from P to +inf in X
            // Edge must span P.y
            if ((v1.y > p.y) != (v2.y > p.y)) {
                // Compute intersection X
                float xInt = v1.x + (p.y - v1.y) * (v2.x - v1.x) / (v2.y - v1.y);
                if (xInt > p.x) {
                    intersections++;
                }
            }
        }
        // Check closing edge? loops usually are closed in segmentsToLoops?
        // segmentsToLoops stores points. If closed loop, first and last might be same or not.
        // Let's assume it might not be explicitly closed in the vector.
        // But usually slicing creates closed loops.
        // Let's check dist(last, first).
        if ((loop.back() - loop.front()).length() > 0.001f) {
            Vec3 v1 = loop.back();
            Vec3 v2 = loop.front();
            if ((v1.y > p.y) != (v2.y > p.y)) {
                float xInt = v1.x + (p.y - v1.y) * (v2.x - v1.x) / (v2.y - v1.y);
                if (xInt > p.x) intersections++;
            }
        }
    }
    return (intersections % 2) != 0;
}

void Mesh::detectHole() {
    showHole = false;
    holes.clear();

    auto processSlice = [&](const std::vector<Segment>& segments, bool isZAxis) -> bool {
        if (segments.empty()) return false;

        auto loops = segmentsToLoops(segments);
        if (loops.empty()) return false;

        int largestLoop = -1;
        float maxAvgR = -1.0f;
        std::vector<float> loopAvgRs;

        for (size_t i = 0; i < loops.size(); ++i) {
             const auto& loop = loops[i];
             size_t n = loop.size();
             if (n < 3) {
                 loopAvgRs.push_back(0.0f);
                 continue;
             }
             Vec3 center = {0,0,0};
             for (const auto& p : loop) center = center + p;
             center = center / (float)n;

             float sumR = 0;
             for (const auto& p : loop) {
                float dx = p.x - center.x;
                float dy = isZAxis ? (p.y - center.y) : (p.z - center.z);
                sumR += std::sqrt(dx*dx + dy*dy);
             }
             float avgR = sumR / n;
             loopAvgRs.push_back(avgR);
             if (avgR > maxAvgR) { maxAvgR = avgR; largestLoop = i; }
        }

        bool foundAny = false;
        for (size_t i = 0; i < loops.size(); ++i) {
            if (i == largestLoop && loops.size() > 1) continue;
            const auto& loop = loops[i];
            size_t n = loop.size();
            if (n < 6) continue;

            Vec3 center = {0,0,0};
            for (const auto& p : loop) center = center + p;
            center = center / (float)n;

            float sumR = 0;
            float maxR = -1e9;
            float minR = 1e9;
            for (const auto& p : loop) {
                float dx = p.x - center.x;
                float dy = isZAxis ? (p.y - center.y) : (p.z - center.z);
                float r = std::sqrt(dx*dx + dy*dy);
                sumR += r;
                if (r > maxR) maxR = r;
                if (r < minR) minR = r;
            }
            float avgR = sumR / n;
            if (avgR < 0.01f) continue;
            float circularity = (maxR - minR) / avgR;
            if (circularity > 0.25f) continue;

            HoleLine hole;
            Vec3 avg = center;
            if (isZAxis) {
                hole.top = {avg.x, avg.y, bounds.maxZ * 1.5f};
                hole.bottom = {avg.x, avg.y, bounds.minZ * 1.5f};
            } else {
                hole.top = {avg.x, bounds.maxY * 1.5f, avg.z};
                hole.bottom = {avg.x, bounds.minY * 1.5f, avg.z};
            }
            holes.push_back(hole);
            foundAny = true;
        }
        return foundAny;
    };

    auto findHoleSeat = [&](const Vec3& holeCenter, float searchRadius, bool isZAxis) -> float {
        float minR = 1e9f;
        for (const auto& t : triangles) {
            Vec3 verts[3] = {t.v1, t.v2, t.v3};
            for (const auto& v : verts) {
                float dx, dy;
                if (isZAxis) { dx = v.x - holeCenter.x; dy = v.y - holeCenter.y; }
                else { dx = v.x - holeCenter.x; dy = v.z - holeCenter.z; }
                float r = std::sqrt(dx*dx + dy*dy);
                if (r < searchRadius && r > 0.05f) {
                    if (r < minR) minR = r;
                }
            }
        }
        if (minR >= 1e8f) return isZAxis ? bounds.maxZ : bounds.maxY;

        float maxSeatH = -1e9f;
        float tolerance = minR * 0.05f + 0.05f;
        bool found = false;
        for (const auto& t : triangles) {
            Vec3 verts[3] = {t.v1, t.v2, t.v3};
            for (const auto& v : verts) {
                float dx, dy, dh;
                if (isZAxis) { dx = v.x - holeCenter.x; dy = v.y - holeCenter.y; dh = v.z; }
                else { dx = v.x - holeCenter.x; dy = v.z - holeCenter.z; dh = v.y; }
                float r = std::sqrt(dx*dx + dy*dy);
                if (r < searchRadius && r > 0.05f) {
                    if (r <= minR + tolerance) {
                        if (dh > maxSeatH) { maxSeatH = dh; found = true; }
                    }
                }
            }
        }
        if (found) return maxSeatH;
        return isZAxis ? bounds.maxZ : bounds.maxY;
    };

    float zMid = (bounds.minZ + bounds.maxZ) * 0.5f;
    std::vector<Segment> segsZ = sliceMeshZ(triangles, zMid);
    if (processSlice(segsZ, true)) {
        for (auto& hole : holes) {
            float searchR = bounds.width() * 0.1f;
            if (searchR < 5.0f) searchR = 5.0f;
            float seatZ = findHoleSeat(hole.top, searchR, true);
            hole.top.z = seatZ;
        }
        showHole = true;
        std::cout << holes.size() << " hole(s) detected along Z axis." << std::endl;
        return;
    }

    float yMid = (bounds.minY + bounds.maxY) * 0.5f;
    std::vector<Segment> segsY = sliceMeshY(triangles, yMid);
    if (processSlice(segsY, false)) {
        for (auto& hole : holes) {
            float searchR = bounds.width() * 0.1f;
            if (searchR < 5.0f) searchR = 5.0f;
            float seatY = findHoleSeat(hole.top, searchR, false);
            hole.top.y = seatY;
        }
        showHole = true;
        std::cout << holes.size() << " hole(s) detected along Y axis." << std::endl;
        return;
    }
    std::cout << "No hole detected." << std::endl;
}

float Mesh::getHeadBottomZ() const {
    if (triangles.empty()) return bounds.minZ;
    float height = bounds.depth();
    float shaftZoneTop = bounds.minZ + height * 0.2f;
    float maxShaftR = 0.0f;

    for (const auto& t : triangles) {
        Vec3 verts[3] = {t.v1, t.v2, t.v3};
        for (const auto& v : verts) {
            if (v.z <= shaftZoneTop) {
                float r = std::sqrt(std::pow(v.x - center.x, 2) + std::pow(v.y - center.y, 2));
                if (r > maxShaftR) maxShaftR = r;
            }
        }
    }
    if (maxShaftR <= 0.001f) maxShaftR = bounds.width() * 0.25f;

    struct VertexInfo { float z; float r; };
    std::vector<VertexInfo> candidates;
    for (const auto& t : triangles) {
        Vec3 verts[3] = {t.v1, t.v2, t.v3};
        for (const auto& v : verts) {
            if (v.z > shaftZoneTop) {
                float r = std::sqrt(std::pow(v.x - center.x, 2) + std::pow(v.y - center.y, 2));
                candidates.push_back({v.z, r});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const VertexInfo& a, const VertexInfo& b) {
        return a.z > b.z;
    });

    float headThreshold = maxShaftR * 1.8f;
    float minHeadZ = bounds.maxZ;
    bool foundHead = false;
    for (const auto& v : candidates) {
        if (v.r > headThreshold) {
            if (v.z < minHeadZ) minHeadZ = v.z;
            foundHead = true;
        }
    }
    return foundHead ? minHeadZ : bounds.maxZ;
}

float Mesh::computeDropDistance(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const {
    // Legacy support or fallback if needed
    // Replaced by findGravitySeating for primary logic
    return 0.0f;
}

bool Mesh::checkCollision(const Mesh& other) const {
    // "This" is Base Mesh (Static)
    // "Other" is Screw Mesh (Moving/Test Position)
    // "this->triangles" are in Base Raw Local Space (relative to Base Center if loaded, but actually raw coords from file).
    // The visual transform of "This" is: T(this->pos) * R(this->rot) * T(-this->center).
    // We need to compare "Other" vertices against "This" triangles in the SAME space.
    // Strategy: Transform "Other" vertices to World, then Inverse Transform them to "This" Raw Local Space.

    // 1. Helpers for Rotation
    auto rotateX = [](Vec3 v, float angle) {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x, v.y * c - v.z * s, v.y * s + v.z * c};
    };
    auto rotateY = [](Vec3 v, float angle) {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x * c + v.z * s, v.y, -v.x * s + v.z * c};
    };
    auto rotateZ = [](Vec3 v, float angle) {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x * c - v.y * s, v.x * s + v.y * c, v.z};
    };

    // 2. Compute Transform Params for Other (to World)
    Vec3 otherC = other.center;
    Vec3 otherP = other.positionOffset;
    Vec3 otherR = other.rotation;

    // 3. Compute Inverse Transform Params for This (World to Raw Local)
    // Forward: v_world = R * (v_raw - center) + pos
    // Inverse: v_raw = invR * (v_world - pos) + center
    Vec3 thisC = this->center;
    Vec3 thisP = this->positionOffset;
    Vec3 thisR = this->rotation;

    // 4. Compute Other's AABB in "This" Raw Local Space
    // We take corners of Other's Local AABB, transform to World, then to This Raw Local.
    float minX = other.bounds.minX, maxX = other.bounds.maxX;
    float minY = other.bounds.minY, maxY = other.bounds.maxY;
    float minZ = other.bounds.minZ, maxZ = other.bounds.maxZ;

    std::vector<Vec3> corners = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, maxY, minZ}, {minX, maxY, minZ},
        {minX, minY, maxZ}, {maxX, minY, maxZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ}
    };

    BoundingBox testBox;
    for(auto& v : corners) {
        // Local Other -> World
        Vec3 t = v - otherC;
        t = rotateX(t, otherR.x); t = rotateY(t, otherR.y); t = rotateZ(t, otherR.z);
        t = t + otherP;

        // World -> Raw Local This
        t = t - thisP;
        // Inverse Rotate: invRz * invRy * invRx (Order matters: reverse of forward)
        // Forward is Rx * Ry * Rz (usually, but draw code does Rx, Ry, Rz glRotate calls)
        // glRotate(x), glRotate(y), glRotate(z) implies M = Rz * Ry * Rx? Or Rx * Ry * Rz?
        // OpenGL matrix stack multiplies on the right: Current = Current * New.
        // Identity * T * Rx * Ry * Rz.
        // So v_transformed = T * Rx * Ry * Rz * v.
        // Inverse = inv(Rz) * inv(Ry) * inv(Rx) * inv(T).
        // My helper functions apply rotation to a vector.
        // Forward: rotateX, then Y, then Z. (Matches Rx * Ry * Rz logic if applied sequentially)
        // Inverse: rotateZ(-angle), then Y, then X.
        t = rotateZ(t, -thisR.z);
        t = rotateY(t, -thisR.y);
        t = rotateX(t, -thisR.x);
        t = t + thisC;

        testBox.minX = std::min(testBox.minX, t.x);
        testBox.maxX = std::max(testBox.maxX, t.x);
        testBox.minY = std::min(testBox.minY, t.y);
        testBox.maxY = std::max(testBox.maxY, t.y);
        testBox.minZ = std::min(testBox.minZ, t.z);
        testBox.maxZ = std::max(testBox.maxZ, t.z);
    }

    // Expand test box slightly
    testBox.minX -= 0.1f; testBox.maxX += 0.1f;
    testBox.minY -= 0.1f; testBox.maxY += 0.1f;
    testBox.minZ -= 0.1f; testBox.maxZ += 0.1f;

    // 5. Collect Candidates from This (Raw Space)
    std::vector<Triangle> candidates;
    for(const auto& t : triangles) {
        float tMinX = std::min({t.v1.x, t.v2.x, t.v3.x});
        float tMaxX = std::max({t.v1.x, t.v2.x, t.v3.x});
        if (tMaxX < testBox.minX || tMinX > testBox.maxX) continue;

        float tMinY = std::min({t.v1.y, t.v2.y, t.v3.y});
        float tMaxY = std::max({t.v1.y, t.v2.y, t.v3.y});
        if (tMaxY < testBox.minY || tMinY > testBox.maxY) continue;

        float tMinZ = std::min({t.v1.z, t.v2.z, t.v3.z});
        float tMaxZ = std::max({t.v1.z, t.v2.z, t.v3.z});
        if (tMaxZ < testBox.minZ || tMinZ > testBox.maxZ) continue;

        candidates.push_back(t);
    }

    if (candidates.empty()) return false;

    // 6. Check Triangles
    for(const auto& t : other.triangles) {
        // Transform t to 'this' Raw Local Space
        Triangle tTransformed;
        Vec3 verts[3] = {t.v1, t.v2, t.v3};
        Vec3 tv[3];
        for(int i = 0; i < 3; ++i) {
            // Local Other -> World
            Vec3 v = verts[i] - otherC;
            v = rotateX(v, otherR.x); v = rotateY(v, otherR.y); v = rotateZ(v, otherR.z);
            v = v + otherP;

            // World -> Raw Local This
            v = v - thisP;
            v = rotateZ(v, -thisR.z);
            v = rotateY(v, -thisR.y);
            v = rotateX(v, -thisR.x);
            v = v + thisC;
            tv[i] = v;
        }
        tTransformed.v1 = tv[0];
        tTransformed.v2 = tv[1];
        tTransformed.v3 = tv[2];

        // Check against candidates (Raw Space)
        for(const auto& ct : candidates) {
            if (checkTriTri(ct, tTransformed)) return true;
        }
    }

    return false;
}

Vec3 Mesh::getAlignedPosition(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const {
    Vec3 startPos;
    Vec3 basePos = this->positionOffset;
    Vec3 baseCenter = this->center;
    Vec3 worldHoleTop = holeTop - baseCenter + basePos;

    startPos.x = worldHoleTop.x;
    startPos.y = worldHoleTop.y;

    if (isZAxis) {
        startPos.z = worldHoleTop.z - screw.bounds.minZ + screw.center.z;
    } else {
        startPos.y = worldHoleTop.y - (screw.bounds.minZ - screw.center.z);
        startPos.z = worldHoleTop.z;
    }
    return startPos;
}

Vec3 Mesh::findGravitySeating(const Mesh& screw, const Vec3& holeTop, bool isZAxis) const {
    Mesh testScrew = screw;
    testScrew.positionOffset = getAlignedPosition(screw, holeTop, isZAxis);

    Vec3 dir = isZAxis ? Vec3{0, 0, -1} : Vec3{0, -1, 0};
    float step = isZAxis ? this->bounds.depth() : this->bounds.height();

    for (int i = 0; i < 10; ++i) {
        Vec3 move = dir * step;
        testScrew.positionOffset = testScrew.positionOffset + move;

        if (checkCollision(testScrew)) {
            testScrew.positionOffset = testScrew.positionOffset - move;
        }
        step = step * 0.5f;
    }

    return testScrew.positionOffset;
}

void Mesh::computeVertexNormals() {
    // Intentionally left blank or implemented locally in generateHollow
}

Mesh Mesh::generateHollow(float wallThickness, float bottomThick, float topThick) const {
    Mesh hollowMesh;
    hollowMesh.active = true;
    hollowMesh.color = color;
    hollowMesh.alpha = 1.0f;
    hollowMesh.positionOffset = positionOffset;
    hollowMesh.rotation = rotation;
    hollowMesh.center = center;
    hollowMesh.bounds = bounds;

    hollowMesh.triangles = triangles; // Copy outer shell

    std::map<Vec3, Vec3> vertexNormalsMap;
    for (const auto& t : triangles) {
        vertexNormalsMap[t.v1] = vertexNormalsMap[t.v1] + t.normal;
        vertexNormalsMap[t.v2] = vertexNormalsMap[t.v2] + t.normal;
        vertexNormalsMap[t.v3] = vertexNormalsMap[t.v3] + t.normal;
    }
    for (auto& pair : vertexNormalsMap) pair.second = pair.second.normalize();

    float minZ = bounds.minZ + bottomThick;
    float maxZ = bounds.maxZ - topThick;

    for (const auto& t : triangles) {
        Triangle innerT;
        innerT.normal = t.normal * -1.0f;
        Vec3 verts[3] = {t.v1, t.v2, t.v3};
        Vec3 newVerts[3];
        for (int i = 0; i < 3; ++i) {
            Vec3 n = vertexNormalsMap[verts[i]];
            Vec3 p = verts[i] - (n * wallThickness);
            if (p.z < minZ) p.z = minZ;
            if (p.z > maxZ) p.z = maxZ;
            newVerts[i] = p;
        }
        innerT.v1 = newVerts[0];
        innerT.v2 = newVerts[2];
        innerT.v3 = newVerts[1];
        hollowMesh.triangles.push_back(innerT);
    }
    return hollowMesh;
}

Mesh Mesh::generateInnerShell(float wallThickness, float bottomThick, float topThick) const {
    Mesh innerMesh;
    innerMesh.active = true;
    innerMesh.color = color;
    innerMesh.alpha = 1.0f;
    innerMesh.positionOffset = positionOffset;
    innerMesh.rotation = rotation;
    innerMesh.center = center;
    innerMesh.bounds = bounds;
    // Bounds might need adjustment but using outer bounds is safe for containers

    std::map<Vec3, Vec3> vertexNormalsMap;
    for (const auto& t : triangles) {
        vertexNormalsMap[t.v1] = vertexNormalsMap[t.v1] + t.normal;
        vertexNormalsMap[t.v2] = vertexNormalsMap[t.v2] + t.normal;
        vertexNormalsMap[t.v3] = vertexNormalsMap[t.v3] + t.normal;
    }
    for (auto& pair : vertexNormalsMap) pair.second = pair.second.normalize();

    float minZ = bounds.minZ + bottomThick;
    float maxZ = bounds.maxZ - topThick;

    for (const auto& t : triangles) {
        Triangle innerT;
        // Inner shell normals should point INWARD for the shell itself
        // But for Point-in-Mesh checks we usually want normals pointing Out of the solid volume.
        // The "solid volume" of the inner shell is the void.
        // If we want to check if a point is INSIDE the void, we treat the void as a solid.
        // So we keep the winding such that normals point towards the center of the void.
        // Original mesh normals point OUT. Inner shell vertices are shifted IN.
        // If we use reversed winding (like generateHollow), normals point IN (towards void center).
        // This makes the void "Solid".

        innerT.normal = t.normal * -1.0f;
        Vec3 verts[3] = {t.v1, t.v2, t.v3};
        Vec3 newVerts[3];

        for (int i = 0; i < 3; ++i) {
            Vec3 n = vertexNormalsMap[verts[i]];
            Vec3 p = verts[i] - (n * wallThickness);
            if (p.z < minZ) p.z = minZ;
            if (p.z > maxZ) p.z = maxZ;
            newVerts[i] = p;
        }

        // Reversed winding: v1, v3, v2
        innerT.v1 = newVerts[0];
        innerT.v2 = newVerts[2];
        innerT.v3 = newVerts[1];

        // Update bounds for inner mesh specifically
        innerMesh.updateBounds(newVerts[0]);
        innerMesh.updateBounds(newVerts[1]);
        innerMesh.updateBounds(newVerts[2]);

        innerMesh.triangles.push_back(innerT);
    }
    return innerMesh;
}

std::vector<Segment> clipLineToLoops(const Vec3& origin, const Vec3& dir, const std::vector<std::vector<Vec3>>& loops) {
    std::vector<Segment> result;
    if (loops.empty()) return result;

    std::vector<float> tValues;

    auto cross2D = [](const Vec3& a, const Vec3& b) {
        return a.x * b.y - a.y * b.x;
    };

    for (const auto& loop : loops) {
        if (loop.size() < 2) continue;
        for (size_t i = 0; i < loop.size(); ++i) {
            Vec3 A, B;
            if (i < loop.size() - 1) {
                 A = loop[i]; B = loop[i+1];
            } else {
                 A = loop.back(); B = loop.front();
                 if ((A - B).length() < 0.001f) continue;
            }

            Vec3 V = B - A;
            float det = cross2D(V, dir);

            if (std::abs(det) > 1e-6f) {
                float t = cross2D(V, A - origin) / det;
                float u = cross2D(dir, A - origin) / det;

                if (u >= 0.0f && u <= 1.0f) {
                    tValues.push_back(t);
                }
            }
        }
    }

    if (tValues.empty()) return result;

    std::sort(tValues.begin(), tValues.end());

    tValues.erase(std::unique(tValues.begin(), tValues.end(), [](float a, float b){
        return std::abs(a - b) < 1e-4f;
    }), tValues.end());

    for (size_t i = 0; i + 1 < tValues.size(); ++i) {
        float t1 = tValues[i];
        float t2 = tValues[i+1];
        float midT = (t1 + t2) * 0.5f;
        Vec3 midP = origin + dir * midT;

        if (isPointInPolygon(midP, loops)) {
            result.push_back({origin + dir * t1, origin + dir * t2});
        }
    }
    return result;
}

Mesh Mesh::generateInfill(float nozzleWidth, float layerHeight, float infillPercent) const {
    Mesh infillMesh;
    infillMesh.active = true;
    infillMesh.color = {1.0f, 0.5f, 0.0f}; // Orange infill
    infillMesh.alpha = 1.0f;
    infillMesh.positionOffset = positionOffset;
    infillMesh.rotation = rotation;
    infillMesh.center = center;
    infillMesh.bounds = bounds;

    if (infillPercent <= 0.01f) return infillMesh;

    // Triangle Infill Parameters
    float spacing = nozzleWidth * (100.0f / infillPercent);
    if (spacing < nozzleWidth) spacing = nozzleWidth;

    // Directions: 0, 60, 120 degrees
    std::vector<Vec3> directions;
    directions.push_back({1.0f, 0.0f, 0.0f}); // 0 deg
    directions.push_back({0.5f, std::sqrt(3.0f)/2.0f, 0.0f}); // 60 deg
    directions.push_back({-0.5f, std::sqrt(3.0f)/2.0f, 0.0f}); // 120 deg

    std::vector<Vec3> normals; // Normals to lines (rotated 90 deg from dir)
    for (const auto& d : directions) {
        normals.push_back({-d.y, d.x, 0.0f});
    }

    // Bounding box corners
    std::vector<Vec3> corners;
    corners.push_back({bounds.minX, bounds.minY, 0});
    corners.push_back({bounds.maxX, bounds.minY, 0});
    corners.push_back({bounds.maxX, bounds.maxY, 0});
    corners.push_back({bounds.minX, bounds.maxY, 0});

    std::vector<float> zLevels;
    for (float z = bounds.minZ; z < bounds.maxZ; z += layerHeight) {
        zLevels.push_back(z);
    }

    // Parallel Processing
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > 1) numThreads--;

    size_t totalLayers = zLevels.size();
    size_t layersPerThread = (totalLayers + numThreads - 1) / numThreads;

    auto processLayerRange = [&](size_t startIdx, size_t endIdx) -> std::vector<Triangle> {
        std::vector<Triangle> localTriangles;

        for (size_t l = startIdx; l < endIdx; ++l) {
            float z = zLevels[l];

            auto boundarySegments = sliceMeshZ(triangles, z);
            auto boundaryLoops = segmentsToLoops(boundarySegments);
            if (boundaryLoops.empty()) continue;

            for (int dirIdx = 0; dirIdx < 3; ++dirIdx) {
                Vec3 D = directions[dirIdx];
                Vec3 N = normals[dirIdx];

                float minProj = 1e9f, maxProj = -1e9f;
                for (const auto& c : corners) {
                    float p = c.x * N.x + c.y * N.y;
                    if (p < minProj) minProj = p;
                    if (p > maxProj) maxProj = p;
                }

                // Snap to grid
                float firstLineOffset = std::ceil(minProj / spacing) * spacing;

                for (float offset = firstLineOffset; offset <= maxProj; offset += spacing) {
                    // Line equation: P . N = offset
                    // O = offset * N
                    Vec3 O = {offset * N.x, offset * N.y, z};

                    std::vector<Segment> segments = clipLineToLoops(O, D, boundaryLoops);

                    for (const auto& seg : segments) {
                        Vec3 p1 = seg.p1;
                        Vec3 p2 = seg.p2;

                        if ((p1 - p2).length() < 0.001f) continue;

                        Vec3 segD = p2 - p1;
                        Vec3 segN = {-segD.y, segD.x, 0};
                        segN = segN.normalize();
                        Vec3 thickOffset = segN * (nozzleWidth * 0.5f);

                        Vec3 b1 = p1 - thickOffset; Vec3 b2 = p1 + thickOffset;
                        Vec3 b3 = p2 + thickOffset; Vec3 b4 = p2 - thickOffset;
                        Vec3 t1 = b1; t1.z += layerHeight;
                        Vec3 t2 = b2; t2.z += layerHeight;
                        Vec3 t3 = b3; t3.z += layerHeight;
                        Vec3 t4 = b4; t4.z += layerHeight;

                        auto addQuad = [&](Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, Vec3 n) {
                            Triangle tri1 = {n, v1, v2, v3};
                            Triangle tri2 = {n, v1, v3, v4};
                            localTriangles.push_back(tri1);
                            localTriangles.push_back(tri2);
                        };

                        addQuad(b2, b1, t1, t2, -segD.normalize());
                        addQuad(b3, b2, t2, t3, segN);
                        addQuad(b4, b3, t3, t4, segD.normalize());
                        addQuad(b1, b4, t4, t1, segN * -1.0f);
                        addQuad(t1, t4, t3, t2, {0,0,1});
                        addQuad(b2, b3, b4, b1, {0,0,-1});
                    }
                }
            }
        }
        return localTriangles;
    };

    std::vector<std::future<std::vector<Triangle>>> futures;
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t start = t * layersPerThread;
        size_t end = std::min(start + layersPerThread, totalLayers);
        if (start < end) {
            futures.push_back(std::async(std::launch::async, processLayerRange, start, end));
        }
    }

    for (auto& f : futures) {
        std::vector<Triangle> part = f.get();
        infillMesh.triangles.insert(infillMesh.triangles.end(), part.begin(), part.end());
    }

    infillMesh.updateBounds(center);
    return infillMesh;
}

void Mesh::initVertexColors(const Vec3& c) {
    vertexColors.resize(triangles.size() * 3);
    for (auto& vc : vertexColors) vc = c;
}

void Mesh::paintVertices(const Vec3& modelHitPoint, float radius, const Vec3& paintColor) {
    if (vertexColors.empty()) {
        initVertexColors(this->color);
    }

    float r2 = radius * radius;
    size_t count = triangles.size();

    for (size_t i = 0; i < count; ++i) {
        const auto& t = triangles[i];

        // Check dist squared for each vertex
        Vec3 d1 = t.v1 - modelHitPoint;
        if (d1.dot(d1) <= r2) vertexColors[3*i] = paintColor;

        Vec3 d2 = t.v2 - modelHitPoint;
        if (d2.dot(d2) <= r2) vertexColors[3*i+1] = paintColor;

        Vec3 d3 = t.v3 - modelHitPoint;
        if (d3.dot(d3) <= r2) vertexColors[3*i+2] = paintColor;
    }
}

bool Mesh::intersectRay(const Vec3& rayOrigin, const Vec3& rayDir, Vec3& hitPoint, float& tMin) const {
    tMin = 1e9f;
    bool hit = false;

    for (const auto& t : triangles) {
        // Moller-Trumbore intersection
        const float kEpsilon = 1e-8;
        Vec3 v0 = t.v1;
        Vec3 v1 = t.v2;
        Vec3 v2 = t.v3;

        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = rayDir.cross(edge2);
        float a = edge1.dot(h);

        if (a > -kEpsilon && a < kEpsilon) continue; // Parallel

        float f = 1.0f / a;
        Vec3 s = rayOrigin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) continue;

        Vec3 q = s.cross(edge1);
        float v = f * rayDir.dot(q);
        if (v < 0.0f || u + v > 1.0f) continue;

        float tVal = f * edge2.dot(q);

        if (tVal > kEpsilon && tVal < tMin) {
            tMin = tVal;
            hitPoint = rayOrigin + rayDir * tVal;
            hit = true;
        }
    }
    return hit;
}

void Mesh::refineMesh(float maxArea) {
    if (!active || triangles.empty()) return;

    bool hasVC = !vertexColors.empty();
    if (hasVC && vertexColors.size() != triangles.size() * 3) {
        // Mismatch safety
        vertexColors.clear();
        hasVC = false;
    }

    struct Task {
        Triangle t;
        Vec3 c1, c2, c3;
    };

    std::vector<Task> stack;
    stack.reserve(triangles.size() * 2);

    for (size_t i = 0; i < triangles.size(); ++i) {
        Task task;
        task.t = triangles[i];
        if (hasVC) {
            task.c1 = vertexColors[3*i];
            task.c2 = vertexColors[3*i+1];
            task.c3 = vertexColors[3*i+2];
        }
        stack.push_back(task);
    }

    // Clear existing to rebuild
    triangles.clear();
    vertexColors.clear();

    while (!stack.empty()) {
        Task task = stack.back();
        stack.pop_back();

        // Calculate Area
        Vec3 e1 = task.t.v2 - task.t.v1;
        Vec3 e2 = task.t.v3 - task.t.v1;
        Vec3 cp = e1.cross(e2);
        float area = 0.5f * cp.length();

        if (area > maxArea) {
            // Split into 4
            Vec3 m1 = (task.t.v1 + task.t.v2) * 0.5f; // v1-v2
            Vec3 m2 = (task.t.v2 + task.t.v3) * 0.5f; // v2-v3
            Vec3 m3 = (task.t.v3 + task.t.v1) * 0.5f; // v3-v1

            // Interpolate colors
            Vec3 cm1 = {0,0,0}, cm2 = {0,0,0}, cm3 = {0,0,0};
            if (hasVC) {
                cm1 = (task.c1 + task.c2) * 0.5f;
                cm2 = (task.c2 + task.c3) * 0.5f;
                cm3 = (task.c3 + task.c1) * 0.5f;
            }

            // Push 4 new triangles
            // T1: v1, m1, m3
            Task t1;
            t1.t = {task.t.normal, task.t.v1, m1, m3};
            t1.c1 = task.c1; t1.c2 = cm1; t1.c3 = cm3;
            stack.push_back(t1);

            // T2: m1, v2, m2
            Task t2;
            t2.t = {task.t.normal, m1, task.t.v2, m2};
            t2.c1 = cm1; t2.c2 = task.c2; t2.c3 = cm2;
            stack.push_back(t2);

            // T3: m3, m2, v3
            Task t3;
            t3.t = {task.t.normal, m3, m2, task.t.v3};
            t3.c1 = cm3; t3.c2 = cm2; t3.c3 = task.c3;
            stack.push_back(t3);

            // T4: m1, m2, m3
            Task t4;
            t4.t = {task.t.normal, m1, m2, m3};
            t4.c1 = cm1; t4.c2 = cm2; t4.c3 = cm3;
            stack.push_back(t4);

        } else {
            triangles.push_back(task.t);
            if (hasVC) {
                vertexColors.push_back(task.c1);
                vertexColors.push_back(task.c2);
                vertexColors.push_back(task.c3);
            }
        }
    }
}

void Mesh::addMesh(const Mesh& other) {
    if (!other.active) return;

    // Check if we need to upgrade to Vertex Colors
    bool needsVC = !vertexColors.empty() || !other.vertexColors.empty();

    // Also if base colors differ, we should use VC to preserve them
    if (!needsVC && (color.x != other.color.x || color.y != other.color.y || color.z != other.color.z)) {
        needsVC = true;
    }

    if (needsVC) {
        if (vertexColors.empty()) {
            initVertexColors(color);
        }
    }

    // Append Triangles
    triangles.insert(triangles.end(), other.triangles.begin(), other.triangles.end());

    // Append Vertex Colors
    if (needsVC) {
        std::vector<Vec3> otherColors = other.vertexColors;
        if (otherColors.empty()) {
            otherColors.resize(other.triangles.size() * 3, other.color);
        }
        vertexColors.insert(vertexColors.end(), otherColors.begin(), otherColors.end());
    }

    // Append Vertex Normals if they exist in source (and target?)
    // Note: computeVertexNormals might need to be re-run if we want smooth shading across the join,
    // but here we just append what we have.
    // If target has normals but source doesn't, or vice-versa, we might have issues.
    // But `Mesh` struct doesn't seem to strictly enforce normals presence for drawing (uses triangle face normals if glNormal3f called with triangle data).
    // Wait, draw() uses `t.normal`. `vertexNormals` is a separate vector, seemingly for `generateHollow` or smooth shading if used.
    // `draw()` does NOT use `vertexNormals` vector directly, it uses `t.normal` from Triangle struct.
    // So `triangles` insert is sufficient for drawing.
    // `vertexNormals` member is used in `generateHollow` but not in `draw`.
    // So we are safe unless we plan to generate hollow AGAIN from this combined mesh.

    // Update bounds
    updateBounds({other.bounds.minX, other.bounds.minY, other.bounds.minZ});
    updateBounds({other.bounds.maxX, other.bounds.maxY, other.bounds.maxZ});
}

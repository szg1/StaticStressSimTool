#ifndef FEA_SOLVER_H
#define FEA_SOLVER_H

#include "mesh.h"
#include <vector>
#include <map>
#include <atomic>
// #include <Accelerate/Accelerate.h> // The Apple Math Library

// =========================================================
// PHYSICS CONSTANTS (PLA/PETG)
// =========================================================
struct Material {
    double E;       // Young's Modulus (Pa) - e.g. 3.5e9 for PLA
    double nu;      // Poisson's Ratio - e.g. 0.36
    double thickness; // Shell thickness (m) - usually your Nozzle Width
};

struct NonlinearMaterial : public Material {
    double yieldStress = 50e6;      // Pa (PLA approx 50-60 MPa)
    double failStress = 60e6;       // Pa (Ultimate strength)
    double hardeningModulus = 0.0;  // Pa (Slope of plastic region)
};

struct ElementState {
    bool failed = false;
    double plasticStrain = 0.0;
    double currentE = 0.0;  // Initialized to Material E
};

// =========================================================
// THE NODE (Degrees of Freedom)
// =========================================================
struct FeaNode {
    int id;           // Solver Index (0 to N-1)
    Vec3 originalPos; // Where it started
    
    // The Solution (Displacement)
    // We solve for 6 DOFs: u_x, u_y, u_z, rot_x, rot_y, rot_z
    double dx = 0, dy = 0, dz = 0;
    double rx = 0, ry = 0, rz = 0;

    // The Force applied to this node
    double fx = 0, fy = 0, fz = 0;
    
    // Is this a fixed anchor? (Screw)
    bool isFixed = false;
};

// =========================================================
// THE SOLVER CLASS
// =========================================================
class FeaSolver {
public:
    // 1. Convert the "Soup of Triangles" into a "Connected Node Mesh"
    // (This welds vertices that share the same location)
    void buildSystem(const Mesh& mesh, const Material& mat);

    // 2. Apply the Load based on Painted Colors
    // paintColor: The color you used in Mesh::paintVertices
    // totalForce: Total Newtons to apply
    // direction: Direction vector of the force
    void applyLoadFromPaint(const Mesh& mesh, const Vec3& paintColor, double totalForce, Vec3 direction);

    // 3. Fix Nodes (Boundary Conditions)
    // Example: Fix everything below z=0.01
    void autoFixGround(float zThreshold = 0.01f);
    
    // 4. Run the Solver (Apple Accelerate)
    bool solve(std::atomic<float>* progress = nullptr);

    // 4b. Run Nonlinear Solver (Incremental Loading + Plasticity)
    bool solveNonlinear(const NonlinearMaterial& mat, int steps = 10, std::atomic<float>* progress = nullptr);

    // 5. Visuals: Get the color map based on Stress
    // Returns a copy of the mesh with vertex colors updated to show stress
    // minStress/maxStress are outputs to help you scale the legend
    void computeStressColorMap(Mesh& targetMesh, float& outMinStress, float& outMaxStress);
    
    void applyDeformation(Mesh& targetMesh, float scale = 1.0f);
private:
    std::vector<FeaNode> nodes;
    std::vector<std::vector<int>> elements; // Each element is 3 Node IDs
    std::vector<ElementState> elementStates; // State for each element
    Material material;

    // Helper: Map Vec3 to Node Index to weld vertices
    int getOrCreateNode(const Vec3& p, float tolerance = 0.0001f);
    std::map<Vec3, int> vertexCache;

    // Helper: Compute Von Mises stress for an element
    double computeElementStress(int elemIdx, const std::vector<FeaNode>& currentNodes, const Material& mat);

    // Helper: Update plasticity/failure state
    void updateElementStates(const NonlinearMaterial& mat);
};

#endif
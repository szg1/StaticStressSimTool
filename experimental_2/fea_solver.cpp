#include "fea_solver.h"
#include <iostream>
#include <cmath>
#include <set>
#include <algorithm>
#include <vector>
#include <map>

// =========================================================
// TINY LINEAR ALGEBRA HELPER
// =========================================================
namespace LinAlg {
    void computeMembraneStiffness(double E, double nu, double h, double area, 
                                  double x2, double x3, double y3, 
                                  double K[6][6]) 
    {
        double f = (E * h) / (1.0 - nu * nu);
        double D[3][3] = {
            {1,  nu, 0},
            {nu, 1,  0},
            {0,  0,  (1.0-nu)/2.0}
        };

        double y23 = y3 - 0;
        double x32 = x2 - x3;
        double y31 = 0 - y3;
        double x13 = x3 - 0;
        double y12 = 0 - 0;
        double x21 = 0 - x2;

        double B[3][6];
        double inv2A = 1.0 / (2.0 * area);

        B[0][0]=y23*inv2A; B[0][1]=0;        B[0][2]=y31*inv2A; B[0][3]=0;        B[0][4]=y12*inv2A; B[0][5]=0;
        B[1][0]=0;         B[1][1]=x32*inv2A;B[1][2]=0;         B[1][3]=x13*inv2A;B[1][4]=0;         B[1][5]=x21*inv2A;
        B[2][0]=x32*inv2A; B[2][1]=y23*inv2A;B[2][2]=x13*inv2A; B[2][3]=y31*inv2A;B[2][4]=x21*inv2A; B[2][5]=y12*inv2A;

        for(int i=0; i<6; i++) {
            for(int j=0; j<6; j++) {
                double val = 0.0;
                for(int k=0; k<3; k++) {
                    for(int l=0; l<3; l++) {
                        val += B[k][i] * D[k][l] * B[l][j];
                    }
                }
                K[i][j] = val * area;
            }
        }
    }
}

// =========================================================
// CUSTOM SOLVER: PRECONDITIONED CONJUGATE GRADIENT (PCG)
// =========================================================
bool solveCG(int n, 
             const std::vector<int>& rowPtr, 
             const std::vector<int>& colInd, 
             const std::vector<double>& val, 
             const std::vector<double>& b, 
             const std::vector<double>& diagonal, 
             std::vector<double>& x,
             std::atomic<float>* progress) // NEW PARAM
{
    std::vector<double> r = b; 
    std::vector<double> z(n);  
    
    for(int i=0; i<n; i++) z[i] = (std::abs(diagonal[i]) > 1e-20) ? r[i] / diagonal[i] : r[i];

    std::vector<double> p = z; 
    std::vector<double> Ap(n);
    
    double rz_old = 0.0;
    for(int i=0; i<n; i++) rz_old += r[i] * z[i];

    // Calculate initial error for progress tracking
    double initial_r_norm = 0.0;
    for(int i=0; i<n; i++) initial_r_norm += r[i] * r[i];
    double initial_err = std::sqrt(initial_r_norm);
    if (initial_err < 1e-9) initial_err = 1e-9;

    std::cout << "  PCG Solver initialized." << std::endl;

    int max_iter = 3000; 
    double tolerance = 1e-5; 

    for(int k=0; k<max_iter; k++) {
        // Matrix-Vector Multiply
        for(int i=0; i<n; i++) {
            double sum = 0.0;
            for(int idx = rowPtr[i]; idx < rowPtr[i+1]; idx++) sum += val[idx] * p[colInd[idx]];
            Ap[i] = sum;
        }

        double pAp = 0.0;
        for(int i=0; i<n; i++) pAp += p[i] * Ap[i];
        
        if (std::abs(pAp) < 1e-20) return false;

        double alpha = rz_old / pAp;
        double r_norm = 0.0;
        
        for(int i=0; i<n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            r_norm += r[i] * r[i];
        }

        double err = std::sqrt(r_norm);
        
        // Smarter progress update based on error reduction
        if (progress && (k%20)==0) {
             // Normalized error heuristic
             double log_current = std::log10(err + 1e-20);
             double log_initial = std::log10(initial_err + 1e-20);
             double log_target = std::log10(tolerance);

             float solver_prog = 0.0f;
             if (std::abs(log_target - log_initial) > 1e-9) {
                 solver_prog = (float)((log_initial - log_current) / (log_initial - log_target));
             }

             // Clamp solver_prog to 0..1
             if (solver_prog < 0.0f) solver_prog = 0.0f;
             if (solver_prog > 1.0f) solver_prog = 1.0f;

             // Map to 0.2 -> 0.9 range
             float global_prog = 0.2f + 0.7f * solver_prog;

             // Ensure monotonicity to prevent jumping
             float currentP = *progress;
             if (global_prog > currentP) {
                 *progress = global_prog;
             }
        }

        if (err < tolerance) {
            if (progress) *progress = 1.0f; // Done
            return true;
        }

        for(int i=0; i<n; i++) z[i] = (std::abs(diagonal[i]) > 1e-20) ? r[i] / diagonal[i] : r[i];

        double rz_new = 0.0;
        for(int i=0; i<n; i++) rz_new += r[i] * z[i];

        double beta = rz_new / rz_old;
        rz_old = rz_new;

        for(int i=0; i<n; i++) p[i] = z[i] + beta * p[i];
    }
    return true; 
}

// =========================================================
// 1. SYSTEM BUILDER
// =========================================================
int FeaSolver::getOrCreateNode(const Vec3& p, float tolerance) {
    auto it = vertexCache.find(p);
    if (it != vertexCache.end()) {
        return it->second;
    }

    int id = (int)nodes.size();
    nodes.push_back({id, p});
    vertexCache[p] = id;
    return id;
}

void FeaSolver::buildSystem(const Mesh& mesh, const Material& mat) {
    nodes.clear();
    elements.clear();
    vertexCache.clear();
    material = mat;

    std::cout << "Building FEA System..." << std::endl;

    for (const auto& tri : mesh.triangles) {
        int n1 = getOrCreateNode(tri.v1);
        int n2 = getOrCreateNode(tri.v2);
        int n3 = getOrCreateNode(tri.v3);

        Vec3 u = tri.v2 - tri.v1;
        Vec3 v = tri.v3 - tri.v1;
        if (u.cross(v).length() < 1e-9) continue;

        elements.push_back({n1, n2, n3});
    }

    std::cout << "  Nodes: " << nodes.size() << std::endl;
    std::cout << "  Elements: " << elements.size() << std::endl;

    // Initialize element states
    elementStates.clear();
    elementStates.resize(elements.size());
    for(auto& es : elementStates) {
        es.currentE = material.E;
        es.plasticStrain = 0.0;
        es.failed = false;
    }
}

// =========================================================
// 2. APPLY LOAD
// =========================================================
void FeaSolver::applyLoadFromPaint(const Mesh& mesh, const Vec3& paintColor, double totalForce, Vec3 direction) {
    if (mesh.vertexColors.empty()) return;

    std::set<int> loadedNodeIndices;
    
    for (size_t i = 0; i < mesh.triangles.size(); ++i) {
        const auto& t = mesh.triangles[i];
        
        Vec3 c1 = mesh.vertexColors[3*i + 0];
        if ((c1 - paintColor).length() < 0.01f) loadedNodeIndices.insert(vertexCache[t.v1]);

        Vec3 c2 = mesh.vertexColors[3*i + 1];
        if ((c2 - paintColor).length() < 0.01f) loadedNodeIndices.insert(vertexCache[t.v2]);

        Vec3 c3 = mesh.vertexColors[3*i + 2];
        if ((c3 - paintColor).length() < 0.01f) loadedNodeIndices.insert(vertexCache[t.v3]);
    }

    if (loadedNodeIndices.empty()) {
        std::cerr << "Warning: No painted vertices found!" << std::endl;
        return;
    }

    double forcePerNode = totalForce / (double)loadedNodeIndices.size();
    Vec3 fVec = direction.normalize() * forcePerNode;

    for (int id : loadedNodeIndices) {
        nodes[id].fx += fVec.x;
        nodes[id].fy += fVec.y;
        nodes[id].fz += fVec.z;
    }
    
    std::cout << "  Applied " << totalForce << "N to " << loadedNodeIndices.size() << " nodes." << std::endl;
}

// =========================================================
// 3. AUTO FIX GROUND
// =========================================================
void FeaSolver::autoFixGround(float zThreshold) {
    int fixedCount = 0;
    float minZ = 1e9;
    for(const auto& n : nodes) if(n.originalPos.z < minZ) minZ = n.originalPos.z;

    for (auto& n : nodes) {
        if (n.originalPos.z <= minZ + zThreshold) {
            n.isFixed = true;
            fixedCount++;
        }
    }
    std::cout << "  Fixed " << fixedCount << " nodes at Z ~ " << minZ << std::endl;
}

// =========================================================
// 4. THE SOLVE (MANUAL CG)
// =========================================================
bool FeaSolver::solve(std::atomic<float>* progress) {
    int nNodes = (int)nodes.size();
    int DOFs = nNodes * 3; 

    if (progress) *progress = 0.0f; // Start

    std::map<long long, double> K_global_map;
    
    auto add_stiffness = [&](int r, int c, double val) {
        if (r < 0 || c < 0) return;
        if (nodes[r/3].isFixed) {
            if (r == c) K_global_map[(long long)r * DOFs + c] += 1.0e15; 
            return;
        }
        if (nodes[c/3].isFixed) return;
        K_global_map[(long long)r * DOFs + c] += val;
    };

    std::cout << "  Assembling Stiffness Matrix..." << std::endl;

    for (size_t elemIdx = 0; elemIdx < elements.size(); ++elemIdx) {
        const auto& elem = elements[elemIdx];

        // NEW: Check if element has failed
        if (elementStates.size() > elemIdx && elementStates[elemIdx].failed) continue;

        // NEW: Use current element stiffness (degraded E)
        double currentE = (elementStates.size() > elemIdx) ? elementStates[elemIdx].currentE : material.E;

        FeaNode& n1 = nodes[elem[0]]; FeaNode& n2 = nodes[elem[1]]; FeaNode& n3 = nodes[elem[2]];
        Vec3 v12 = n2.originalPos - n1.originalPos; Vec3 v13 = n3.originalPos - n1.originalPos;
        Vec3 normal = v12.cross(v13).normalize();
        double area = 0.5 * v12.cross(v13).length();
        if (area < 1e-12) continue;
        Vec3 e1 = v12.normalize(); Vec3 e2 = normal.cross(e1).normalize();
        double x2=v12.length(); double x3=v13.dot(e1), y3=v13.dot(e2);
        double k_local[6][6];
        LinAlg::computeMembraneStiffness(currentE, material.nu, material.thickness, area, x2, x3, y3, k_local);
        double T[3][3] = { {e1.x, e1.y, e1.z}, {e2.x, e2.y, e2.z}, {normal.x, normal.y, normal.z} };
        int node_indices[3] = {elem[0], elem[1], elem[2]};
        for (int i=0; i<3; i++) { 
            for (int j=0; j<3; j++) { 
                double k_sub[2][2] = { {k_local[i*2][j*2], k_local[i*2][j*2+1]}, {k_local[i*2+1][j*2], k_local[i*2+1][j*2+1]} };
                double k_local_3D[3][3] = { {k_sub[0][0], k_sub[0][1], 0}, {k_sub[1][0], k_sub[1][1], 0}, {0,0,0} };
                k_local_3D[2][2] = area * currentE * 1e-4;
                double K_glob_block[3][3] = {0};
                for(int r=0; r<3; r++) for(int c=0; c<3; c++) {
                        double val = 0;
                        for(int k=0; k<3; k++) for(int l=0; l<3; l++) val += T[k][r] * k_local_3D[k][l] * T[l][c];
                        K_glob_block[r][c] = val;
                }
                int row_base = node_indices[i] * 3; int col_base = node_indices[j] * 3;
                for(int r=0; r<3; r++) for(int c=0; c<3; c++) add_stiffness(row_base+r, col_base+c, K_glob_block[r][c]);
            }
        }
    }

    if (progress) *progress = 0.1f; // Assembly done

    std::cout << "  Converting Map to CSR Format..." << std::endl;
    
    std::vector<int> csr_rowPtr;
    std::vector<int> csr_colInd;
    std::vector<double> csr_val;
    std::vector<double> diagonal(DOFs, 1.0);
    
    csr_rowPtr.reserve(DOFs + 1);
    csr_colInd.reserve(K_global_map.size());
    csr_val.reserve(K_global_map.size());
    
    csr_rowPtr.push_back(0);
    int currentRow = 0;
    int nnz_count = 0;

    for(auto const& [key, val] : K_global_map) {
        int r = (int)(key / DOFs);
        int c = (int)(key % DOFs);
        if (r == c) diagonal[r] = val;
        while(currentRow < r) { csr_rowPtr.push_back(nnz_count); currentRow++; }
        csr_colInd.push_back(c);
        csr_val.push_back(val);
        nnz_count++;
    }
    while(currentRow < DOFs) { csr_rowPtr.push_back(nnz_count); currentRow++; }

    std::vector<double> rhs(DOFs, 0.0);
    for(int i=0; i<nNodes; i++) {
        if (!nodes[i].isFixed) {
            rhs[i*3+0] = nodes[i].fx; rhs[i*3+1] = nodes[i].fy; rhs[i*3+2] = nodes[i].fz;
        }
    }

    if (progress) *progress = 0.2f; // Setup done

    std::cout << "  Starting PCG Solver (" << DOFs << " DOFs)..." << std::endl;
    
    std::vector<double> solution(DOFs, 0.0);
    // PASS PROGRESS POINTER
    bool success = solveCG(DOFs, csr_rowPtr, csr_colInd, csr_val, rhs, diagonal, solution, progress);
    
    if (!success) {
        std::cerr << "  Solver Diverged!" << std::endl;
        return false;
    }
    
    if (progress) *progress = 0.9f;

    for(int i=0; i<nNodes; i++) {
        nodes[i].dx = solution[i*3+0];
        nodes[i].dy = solution[i*3+1];
        nodes[i].dz = solution[i*3+2];
    }
    
    if (progress) *progress = 1.0f;
    return true;
}

// =========================================================
// 4b. NONLINEAR SOLVER
// =========================================================
bool FeaSolver::solveNonlinear(const NonlinearMaterial& mat, int steps, std::atomic<float>* progress) {
    std::cout << "Starting Nonlinear Solver (" << steps << " steps)" << std::endl;

    // 1. Store total applied forces
    std::vector<Vec3> totalForces(nodes.size());
    for(size_t i=0; i<nodes.size(); i++) {
        totalForces[i] = { (float)nodes[i].fx, (float)nodes[i].fy, (float)nodes[i].fz };
        // Reset displacements for fresh start
        nodes[i].dx = nodes[i].dy = nodes[i].dz = 0.0;
    }

    // 2. Incremental Loading Loop
    for (int step = 1; step <= steps; step++) {
        double loadFactor = (double)step / (double)steps;
        std::cout << "  Step " << step << "/" << steps << " (Load: " << (loadFactor*100) << "%)" << std::endl;

        // Apply fraction of load
        for(size_t i=0; i<nodes.size(); i++) {
            nodes[i].fx = totalForces[i].x * loadFactor;
            nodes[i].fy = totalForces[i].y * loadFactor;
            nodes[i].fz = totalForces[i].z * loadFactor;
        }

        // Solve Linear System for this step
        // (Note: solve() re-assembles K using current ElementStates)
        if (!solve(nullptr)) {
            std::cerr << "  Step " << step << " failed to converge!" << std::endl;
            return false;
        }

        // Update Material State (Plasticity/Failure)
        updateElementStates(mat);

        if (progress) *progress = (float)step / (float)steps;
    }

    std::cout << "Nonlinear simulation complete." << std::endl;
    return true;
}

double FeaSolver::computeElementStress(int elemIdx, const std::vector<FeaNode>& currentNodes, const Material& mat) {
    // 1. Get Element Nodes
    const auto& elem = elements[elemIdx];
    Vec3 p1 = currentNodes[elem[0]].originalPos;
    Vec3 p2 = currentNodes[elem[1]].originalPos;
    Vec3 p3 = currentNodes[elem[2]].originalPos;

    // Displacements
    Vec3 u1 = { (float)currentNodes[elem[0]].dx, (float)currentNodes[elem[0]].dy, (float)currentNodes[elem[0]].dz };
    Vec3 u2 = { (float)currentNodes[elem[1]].dx, (float)currentNodes[elem[1]].dy, (float)currentNodes[elem[1]].dz };
    Vec3 u3 = { (float)currentNodes[elem[2]].dx, (float)currentNodes[elem[2]].dy, (float)currentNodes[elem[2]].dz };

    // 2. Compute Local Coordinate System (Same as Stiffness Matrix)
    Vec3 v12 = p2 - p1;
    Vec3 v13 = p3 - p1;
    Vec3 normal = v12.cross(v13).normalize();
    Vec3 e1 = v12.normalize();
    Vec3 e2 = normal.cross(e1).normalize();

    // 3. Strain Displacement Matrix (B) - Simplified CST
    // Transform displacements to local 2D system
    // Local coords: x1=0,y1=0. x2=|v12|. x3,y3 from projection.
    double x2 = v12.length();
    double x3 = v13.dot(e1);
    double y3 = v13.dot(e2);
    double area = 0.5 * (x2 * y3); // Area = 0.5 * base * height

    // Local displacements (project global u onto e1, e2)
    double u1x = u1.dot(e1), u1y = u1.dot(e2);
    double u2x = u2.dot(e1), u2y = u2.dot(e2);
    double u3x = u3.dot(e1), u3y = u3.dot(e2);

    // Constant Strain Triangle (CST) Formula
    // eps_x = (1/2A) * [y23*u1x + y31*u2x + y12*u3x]
    // eps_y = (1/2A) * [x32*u1y + x13*u2y + x21*u3y]
    // gam_xy= (1/2A) * [x32*u1x + y23*u1y + x13*u2x + y31*u2y + x21*u3x + y12*u3y]

    double y23 = y3 - 0;
    double y31 = 0 - y3;
    double y12 = 0 - 0;

    double x32 = x2 - x3;
    double x13 = x3 - 0;
    double x21 = 0 - x2;

    double inv2A = 1.0 / (2.0 * area);

    double eps_x = inv2A * (y23*u1x + y31*u2x + y12*u3x);
    double eps_y = inv2A * (x32*u1y + x13*u2y + x21*u3y);
    double gam_xy= inv2A * (x32*u1x + y23*u1y + x13*u2x + y31*u2y + x21*u3x + y12*u3y);

    // 4. Stress Calculation (Plane Stress)
    // sigma = E/(1-nu^2) * [1  nu 0; nu 1 0; 0 0 (1-nu)/2] * strain
    double factor = mat.E / (1.0 - mat.nu * mat.nu);
    double sig_x = factor * (eps_x + mat.nu * eps_y);
    double sig_y = factor * (mat.nu * eps_x + eps_y);
    double tau_xy= factor * ((1.0 - mat.nu) / 2.0) * gam_xy;

    // 5. Von Mises Stress (2D Plane Stress)
    // sigma_vm = sqrt(sig_x^2 - sig_x*sig_y + sig_y^2 + 3*tau_xy^2)
    double vm = std::sqrt(sig_x*sig_x - sig_x*sig_y + sig_y*sig_y + 3.0*tau_xy*tau_xy);

    return vm;
}

void FeaSolver::updateElementStates(const NonlinearMaterial& mat) {
    int failedCount = 0;
    int yieldCount = 0;

    for(size_t i=0; i<elements.size(); i++) {
        if (elementStates[i].failed) continue;

        double vmStress = computeElementStress(i, nodes, mat);

        // 1. Check Failure
        if (vmStress > mat.failStress) {
            elementStates[i].failed = true;
            elementStates[i].currentE = 1e-6; // "Remove" element (negligible stiffness)
            failedCount++;
            continue;
        }

        // 2. Check Yielding (Plasticity)
        // Simple Isotropic Hardening or Stiffness Degradation
        if (vmStress > mat.yieldStress) {
            // E_tangent = E * (yield / stress) roughly approximates perfectly plastic
            // Or use hardening modulus if defined

            // Heuristic: Reduce stiffness to limit stress to yieldStress
            // New E = Old E * (Yield / Current)
            double reduction = mat.yieldStress / vmStress;
            elementStates[i].currentE = mat.E * reduction;

            // Ensure we don't increase stiffness
            if (elementStates[i].currentE > mat.E) elementStates[i].currentE = mat.E;

            yieldCount++;
        } else {
             // Elastic recovery (simplified)
             elementStates[i].currentE = mat.E;
        }
    }

    if (failedCount > 0 || yieldCount > 0) {
        std::cout << "  State Update: " << failedCount << " failed, " << yieldCount << " yielding." << std::endl;
    }
}

// =========================================================
// 5. VISUALIZATION
// =========================================================
void FeaSolver::computeStressColorMap(Mesh& targetMesh, float& outMinStress, float& outMaxStress) {
    outMinStress = 1e9;
    outMaxStress = -1e9;
    
    std::vector<float> disps;
    for(const auto& n : nodes) {
        float d = (float)std::sqrt(n.dx*n.dx + n.dy*n.dy + n.dz*n.dz);
        disps.push_back(d);
        if (d < outMinStress) outMinStress = d;
        if (d > outMaxStress) outMaxStress = d;
    }
    
    if (std::abs(outMaxStress - outMinStress) < 1e-9) outMaxStress = outMinStress + 1.0;

    targetMesh.vertexColors.resize(targetMesh.triangles.size() * 3);
    
    for(size_t i=0; i<targetMesh.triangles.size(); i++) {
        Vec3 verts[3] = {targetMesh.triangles[i].v1, targetMesh.triangles[i].v2, targetMesh.triangles[i].v3};
        
        for(int k=0; k<3; k++) {
            auto it = vertexCache.find(verts[k]);
            float val = 0;
            if (it != vertexCache.end()) {
                val = disps[it->second];
            }
            
            float t = (val - outMinStress) / (outMaxStress - outMinStress);
            Vec3 c;
            c.x = t;        
            c.y = 1.0f - t; 
            c.z = 0.0f;     
            
            targetMesh.vertexColors[3*i + k] = c;
        }
    }
}

// =========================================================
// 6. DEFORMATION
// =========================================================
void FeaSolver::applyDeformation(Mesh& targetMesh, float scale) {
    std::cout << "  Applying Deformation (Scale: " << scale << "x)..." << std::endl;

    for(size_t i=0; i<targetMesh.triangles.size(); i++) {
        // We modify the vertices directly in the mesh
        // Note: Vertices are shared between triangles in the logic, 
        // but stored duplicated in the STL vector.
        // We iterate vertices 1, 2, 3
        
        Vec3* verts[3] = {&targetMesh.triangles[i].v1, &targetMesh.triangles[i].v2, &targetMesh.triangles[i].v3};
        
        for(int k=0; k<3; k++) {
            // Find which FEA node corresponds to this vertex
            // We use the ORIGINAL position to look it up in the cache
            auto it = vertexCache.find(*verts[k]);
            
            if (it != vertexCache.end()) {
                int nodeId = it->second;
                const auto& node = nodes[nodeId];
                
                // Add the displacement vector
                verts[k]->x += node.dx * scale;
                verts[k]->y += node.dy * scale;
                verts[k]->z += node.dz * scale;
            }
        }
    }
    
    // Recalculate bounds so the camera/bounding box updates
    targetMesh.bounds.reset();
    for(const auto& t : targetMesh.triangles) {
        targetMesh.updateBounds(t.v1);
        targetMesh.updateBounds(t.v2);
        targetMesh.updateBounds(t.v3);
    }
    targetMesh.center = {
        targetMesh.bounds.minX + targetMesh.bounds.width()/2.0f,
        targetMesh.bounds.minY + targetMesh.bounds.height()/2.0f,
        targetMesh.bounds.minZ + targetMesh.bounds.depth()/2.0f
    };
}
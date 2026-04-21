/*
 * Simple STL (Stereolithography) Viewer for macOS
 * Uses OpenGL and GLUT (Native macOS Frameworks)
 *
 * COMPILATION INSTRUCTIONS:
 * Open your terminal and run:
 * make
 *
 * USAGE:
 * ./stl_viewer path/to/your/file.stl
 *
 * CONTROLS:
 * Left Click + Drag: Rotate
 * Right Click + Drag (or Up/Down): Zoom
 * Press 'A' or click "Load screw" button to add a second model.
 */

#define GL_SILENCE_DEPRECATION // Silence macOS OpenGL deprecation warnings

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib> // For exit
#include <thread>
#include <atomic>

#ifdef __APPLE__
#include <GLUT/glut.h> // macOS specific GLUT include
#else
#include <GL/glut.h>
#endif

#include "geometry.h"
#include "mesh.h"
#include "button.h"
#include "utils.h"
#include "fea_solver.h"
#include "gravity_anim_state.h"

// --- Globals ---

enum AppProgress
{
    EMPTY,
    MODELLOADED,
    SCREWSLOADED,
    GENERATING_WALLS,
    WALLSGENERATED,
    ANIMATING_WALLS,
    WALLS_DONE,
    GENERATING_INFILL,
    INFILLGENERATED,
    ANIMATING_INFILL,
    REFINING_MESH,
    FULLYSLICED,
    PAINTING,
    ALL_DONE
};

AppProgress currentProgress = EMPTY;
bool autoSlice = false;

std::vector<Mesh> meshes; // 0 = Main, 1 = Screw/Second
float globalMaxDimension = 1.0f;
int selectedMeshIndex = -1;
// Vec3 globalCenter = {0, 0, 0}; // Unused

// Camera interaction
float rotX = 0.0f;
float rotY = 0.0f;
float zoom = -3.0f;
int lastMouseX, lastMouseY;
int mouseState = 0; // 0 = None, 1 = Rotate, 2 = Zoom
int windowW = 800;
int windowH = 600;

// Matrix Cache for Picking
GLdouble cachedModelView[16];
GLdouble cachedProjection[16];
GLint cachedViewport[4];

// Animation State
bool isAnimating = false;
float animProgress = 0.0f;
std::vector<Vec3> animStartPos;
std::vector<Vec3> animTargetPos;
std::vector<GravityAnimationState> gravityAnimStates;
bool isGravityAnimating = false;

// UI State
Button loadBtn;
Button sliceBtn;
Button paintBtn;
Button forceBtn;
Button simBtn;

// Force Input State
bool isForceSetupMode = false;
std::string forceInputBuffer = "";
std::string displayedForceStr = "";

// Slice Setup State
bool isSliceSetupMode = false;
bool sliceParamsCollected = false;
int currentSliceInputIndex = 0;
std::string currentInputBuffer = "";
std::vector<std::string> sliceInputs(4); // nozzle, layer, wall, infill
const char *slicePrompts[] = {
    "Enter Nozzle Width (mm):",
    "Enter Layer Height (mm):",
    "Enter Wall Count:",
    "Enter Infill %:"};

// Slicing Process State
bool isSlicing = false;
int currentSliceLayer = 0;
int totalLayers = 0;
float currentLayerHeight = 0.2f;
Mesh slicedMesh;
Mesh infillMesh;     // New global for gyroid infill
Mesh tempInfillMesh; // For background thread

std::thread infillThread;
std::atomic<bool> infillReady(false);
std::atomic<bool> infillError(false);

bool isSimulating = false; // Flag to block UI
std::atomic<float> simProgress{0.0f}; // Shared progress (0.0 - 1.0)
std::thread simThread; // The background worker
Mesh simResultMesh; // Buffer for the thread to write to
std::string simStatusText = "";
float simMinS, simMaxS; // Result range


// --- Helper Functions ---

// Forward declarations
void processState(int value);
void generateWalls(int value);
void generateInfill(int value);
void simulationWorker(Mesh inputMesh, Material mat, float force, Vec3 dir, Vec3 paintColor);
void checkSimulation(int value);

void enterPaintMode()
{
    if (currentProgress == FULLYSLICED)
    {
        currentProgress = PAINTING;
        glutPostRedisplay();
        std::cout << "Entered Painting Mode. Left click and drag on the body to paint." << std::endl;
    }
    else if (currentProgress == PAINTING)
    {
        currentProgress = FULLYSLICED;
        glutPostRedisplay();
        std::cout << "Exited Painting Mode." << std::endl;
    }
}

void enterForceMode()
{
    isForceSetupMode = true;
    forceInputBuffer = "";
    glutPostRedisplay();
}

void parseForceInput()
{
    std::string s = forceInputBuffer;
    // Remove spaces
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());

    float val = 0.0f;
    bool valid = false;

    if (s.empty())
        return;

    try
    {
        if (s.size() > 2 && s.substr(s.size() - 2) == "kN")
        {
            val = std::stof(s.substr(0, s.size() - 2)) * 1000.0f;
            valid = true;
        }
        else if (s.size() > 1 && s.back() == 'N')
        {
            val = std::stof(s.substr(0, s.size() - 1));
            valid = true;
        }
        else
        {
            // Assume Newtons
            size_t idx;
            val = std::stof(s, &idx);
            // Check if entire string was number (roughly)
            valid = true;
        }
    }
    catch (...)
    {
        valid = false;
    }

    if (valid)
    {
        char buf[128];
        snprintf(buf, 128, "Force: %.2f N", val);
        displayedForceStr = std::string(buf);
    }
    else
    {
        displayedForceStr = "Invalid Force";
    }
}

void paintOnSurface(int x, int y)
{
    if (!slicedMesh.active)
        return;

    // 1. Get Ray from Camera using Cached Matrices
    GLfloat winX, winY;
    GLdouble nearX, nearY, nearZ;
    GLdouble farX, farY, farZ;

    winX = (float)x;
    winY = (float)cachedViewport[3] - (float)y;

    // Get near and far points
    if (gluUnProject(winX, winY, 0.0, cachedModelView, cachedProjection, cachedViewport, &nearX, &nearY, &nearZ) == GL_FALSE)
        return;
    if (gluUnProject(winX, winY, 1.0, cachedModelView, cachedProjection, cachedViewport, &farX, &farY, &farZ) == GL_FALSE)
        return;

    Vec3 rayOrigin = {(float)nearX, (float)nearY, (float)nearZ};
    Vec3 rayEnd = {(float)farX, (float)farY, (float)farZ};
    Vec3 rayDir = (rayEnd - rayOrigin).normalize();

    // 2. Transform Ray to slicedMesh Local Space
    // slicedMesh is drawn with transformations:
    // T(posOffset) * R(rot) * T(-center)
    // So to go World -> Local:
    // T(center) * R(-rot) * T(-posOffset)

    // Apply inverse translations
    Vec3 localOrigin = rayOrigin - slicedMesh.positionOffset;
    Vec3 localDir = rayDir; // Direction not affected by translation

    // Apply inverse rotations.
    // Original Transform: T(pos) * Rx * Ry * Rz * T(-center)
    // We are going from World -> Local.
    // Inverse Transform: T(center) * inv(Rz) * inv(Ry) * inv(Rx) * T(-pos)
    // Note: We already applied T(-pos) above. Now we need inv(Rz)*inv(Ry)*inv(Rx).
    // inv(Rz) means rotate -Z.
    // Order of application to vector v: inv(Rx) applied first, then inv(Ry), then inv(Rz).
    // Wait, if M = Rx * Ry * Rz, then v_world = Rx(Ry(Rz(v_local))).
    // So v_local = inv(Rz)(inv(Ry)(inv(Rx)(v_world))).
    // So we must apply inv(Rx) first, then inv(Ry), then inv(Rz).

    auto rotateX = [](Vec3 v, float angle)
    {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x, v.y * c - v.z * s, v.y * s + v.z * c};
    };
    auto rotateY = [](Vec3 v, float angle)
    {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x * c + v.z * s, v.y, -v.x * s + v.z * c};
    };
    auto rotateZ = [](Vec3 v, float angle)
    {
        float rad = angle * M_PI / 180.0f;
        float c = cos(rad);
        float s = sin(rad);
        return Vec3{v.x * c - v.y * s, v.x * s + v.y * c, v.z};
    };

    // Apply inverse rotations: inv(Rx) -> inv(Ry) -> inv(Rz)
    localOrigin = rotateX(localOrigin, -slicedMesh.rotation.x);
    localOrigin = rotateY(localOrigin, -slicedMesh.rotation.y);
    localOrigin = rotateZ(localOrigin, -slicedMesh.rotation.z);

    localDir = rotateX(localDir, -slicedMesh.rotation.x);
    localDir = rotateY(localDir, -slicedMesh.rotation.y);
    localDir = rotateZ(localDir, -slicedMesh.rotation.z);

    // Apply Center translation (inverse of -center is +center)
    localOrigin = localOrigin + slicedMesh.center;

    // 3. Intersect
    Vec3 hitPoint;
    float t;
    if (slicedMesh.intersectRay(localOrigin, localDir, hitPoint, t))
    {
        // 4. Paint
        // Radius 3mm. Convert 3mm to mesh units?
        // Assuming mesh units are mm.
        slicedMesh.paintVertices(hitPoint, 3.0f, {1.0f, 0.5f, 0.0f}); // Orange
        glutPostRedisplay();
    }
}
void startSimulation() {
    if (isSimulating) return; // Prevent double click
    
    std::cout << "Starting Simulation (Background Thread)..." << std::endl;
    
    // Gather Params
    Material mat;
    mat.E = 3.5e9; mat.nu = 0.36; 
    try {
        float nozzleMM = std::stof(sliceInputs[0]);
        mat.thickness = nozzleMM / 1000.0f;
    } catch (...) { mat.thickness = 0.0004; }

    if (!slicedMesh.active) return;

    float forceVal = 100.0f;
    try {
        std::string s = displayedForceStr;
        size_t colonPos = s.find(':');
        size_t unitPos = s.find('N');
        if (colonPos != std::string::npos) {
            std::string numStr = s.substr(colonPos + 1, unitPos - colonPos - 1);
            forceVal = std::stof(numStr);
        }
    } catch (...) {}

    Vec3 down = {0, 0, -1};
    Vec3 paintColor = {1.0f, 0.5f, 0.0f};

    // UI Updates
    isSimulating = true;
    simProgress = 0.0f;
    simStatusText = "Calculating...";
    
    // Spawn Thread (Pass slicedMesh by VALUE to create a thread-safe copy)
    simThread = std::thread(simulationWorker, slicedMesh, mat, forceVal, down, paintColor);
    
    // Start Polling
    glutTimerFunc(100, checkSimulation, 0);
}
void renderText(int x, int y, const char *text, void *font = GLUT_BITMAP_HELVETICA_18)
{
    glRasterPos2i(x, y);
    while (*text)
    {
        glutBitmapCharacter(font, *text++);
    }
}

int getTextWidth(const char *text, void *font = GLUT_BITMAP_HELVETICA_18)
{
    int width = 0;
    while (*text)
    {
        width += glutBitmapWidth(font, *text++);
    }
    return width;
}

void updateGravitySeating(int value) {
    if (!isGravityAnimating) return;

    bool anyActive = false;
    for (auto& state : gravityAnimStates) {
        if (!state.active || state.finished) continue;
        if (state.meshIndex >= meshes.size()) continue;

        anyActive = true;
        Mesh& screw = meshes[state.meshIndex];

        if (state.phase == 0) { // MOVE
            // Execute Move
            state.lastMove = state.dir * state.step;
            screw.positionOffset = screw.positionOffset + state.lastMove;
            state.phase = 1; // Next is Check
        } else { // CHECK
            // Check Collision
            if (meshes[0].checkCollision(screw)) {
                // Undo
                std::cout << "Collision detected for screw " << state.meshIndex << ", undoing move." << std::endl;
                screw.positionOffset = screw.positionOffset - state.lastMove;
                state.step *= 0.5f;
                state.iteration++;
            }
            // Prepare next iteration
            state.phase = 0; // Back to Move

            if (state.iteration >= state.maxIterations) {
                state.finished = true;
                std::cout << "Gravity seating done for screw " << state.meshIndex << std::endl;
            }
        }
    }

    glutPostRedisplay();

    if (anyActive) {
        glutTimerFunc(50, updateGravitySeating, 0); // ~20fps per step phase (slow enough to see)
    } else {
        isGravityAnimating = false;
        isAnimating = false; // Fully done

        if (currentProgress == MODELLOADED)
        {
            currentProgress = SCREWSLOADED;
        }
        // Check if we need to proceed with slicing
        glutTimerFunc(100, processState, 0);
    }
}

void updateAnimation(int value)
{
    if (!isAnimating)
        return;

    // If we are in gravity phase, delegate
    if (isGravityAnimating) {
        updateGravitySeating(0);
        return;
    }

    animProgress += 0.02f; // Increment progress (approx 50 frames for 1 sec)
    if (animProgress > 1.0f)
        animProgress = 1.0f;

    // Easing function (Quadratic Ease Out)
    float t = animProgress * (2.0f - animProgress);

    for (size_t i = 1; i < meshes.size(); ++i)
    {
        if (i >= animStartPos.size() || i >= animTargetPos.size())
            break;

        Vec3 start = animStartPos[i];
        Vec3 target = animTargetPos[i];

        meshes[i].positionOffset = {
            start.x + (target.x - start.x) * t,
            start.y + (target.y - start.y) * t,
            start.z + (target.z - start.z) * t};
    }

    glutPostRedisplay();

    if (animProgress < 1.0f)
    {
        glutTimerFunc(16, updateAnimation, 0); // ~60 FPS
    }
    else
    {
        // Linear animation done. Check if we have gravity animations pending.
        bool hasGravity = false;
        for(const auto& s : gravityAnimStates) {
            if (s.active && !s.finished) {
                hasGravity = true;
                break;
            }
        }

        if (hasGravity) {
            isGravityAnimating = true;
            glutTimerFunc(50, updateGravitySeating, 0);
        } else {
            isAnimating = false;
            if (currentProgress == MODELLOADED)
            {
                currentProgress = SCREWSLOADED;
            }
            // Check if we need to proceed with slicing
            glutTimerFunc(100, processState, 0);
        }
    }
}

void updateSlicing(int value)
{
    if (!isSlicing)
        return;

    currentSliceLayer++;
    if (currentSliceLayer > totalLayers)
    {
        currentSliceLayer = totalLayers;
        isSlicing = false; // Finished

        // Transition based on current progress
        if (currentProgress == ANIMATING_WALLS)
        {
            currentProgress = WALLS_DONE;
            glutTimerFunc(50, processState, 0);
        }
        else if (currentProgress == ANIMATING_INFILL)
        {
            currentProgress = REFINING_MESH;

            // Combine Infill into Sliced Mesh for unified simulation/painting
            slicedMesh.addMesh(infillMesh);
            infillMesh.active = false; // Hide separate infill
            glutTimerFunc(50, processState, 0);
        }
    }
    else
    {
        glutTimerFunc(100, updateSlicing, 0); // Slower updates to visualize layers
    }
    glutPostRedisplay();
}

void generateWalls(int value)
{
    try
    {
        float nozzle = std::stof(sliceInputs[0]);
        float layerH = std::stof(sliceInputs[1]);
        int walls = std::stoi(sliceInputs[2]);

        float wallThickness = nozzle * walls;
        float bottomThick = layerH * walls;
        float topThick = layerH * walls;

        // 1. Generate the Full Hollow Mesh (Walls)
        slicedMesh = meshes[0].generateHollow(wallThickness, bottomThick, topThick);
        slicedMesh.alpha = 0.5f; // Transparent

        currentProgress = WALLSGENERATED;
        glutPostRedisplay();
        glutTimerFunc(50, processState, 0); // Trigger next step
    }
    catch (...)
    {
        std::cerr << "Error generating walls." << std::endl;
        sliceParamsCollected = false;
        currentProgress = SCREWSLOADED; // Reset to allow retry
    }
}

void backgroundInfillTask()
{
    try
    {
        float nozzle = std::stof(sliceInputs[0]);
        float layerH = std::stof(sliceInputs[1]);
        int walls = std::stoi(sliceInputs[2]);
        float infill = std::stof(sliceInputs[3]);

        float wallThickness = nozzle * walls;
        float bottomThick = layerH * walls;
        float topThick = layerH * walls;

        // Use tempInfillMesh for thread safety
        Mesh innerShell = meshes[0].generateInnerShell(wallThickness, bottomThick, topThick);
        tempInfillMesh = innerShell.generateInfill(nozzle, layerH, infill);

        // Transform match (copy params, safe to read as they don't change during anim)
        tempInfillMesh.positionOffset = meshes[0].positionOffset;
        tempInfillMesh.rotation = meshes[0].rotation;
        tempInfillMesh.center = meshes[0].center;

        infillReady = true;
    }
    catch (...)
    {
        infillError = true;
    }
}

void checkInfillReady(int value)
{
    if (infillReady)
    {
        if (infillThread.joinable())
            infillThread.join();

        // Move temp to global
        infillMesh = tempInfillMesh;

        currentProgress = INFILLGENERATED;
        glutPostRedisplay();
        glutTimerFunc(50, processState, 0);
    }
    else if (infillError)
    {
        if (infillThread.joinable())
            infillThread.join();
        std::cerr << "Infill generation failed in background." << std::endl;
        currentProgress = SCREWSLOADED; // Reset on error
        glutPostRedisplay();
    }
    else
    {
        // Still generating, check again
        glutTimerFunc(100, checkInfillReady, 0);
    }
}

void startSlicingAnimation()
{
    try
    {
        float layerH = std::stof(sliceInputs[1]);
        currentLayerHeight = layerH;
        float height = slicedMesh.bounds.depth();
        totalLayers = (int)(height / layerH);
        currentSliceLayer = 0;
        isSlicing = true;

        // Make original mesh ghost
        meshes[0].alpha = 0.1f;

        glutTimerFunc(100, updateSlicing, 0);
    }
    catch (...)
    {
    }
}

void processState(int value)
{
    if (!sliceParamsCollected)
        return;

    if (currentProgress == MODELLOADED)
    {
        if (!isAnimating)
        {
            currentProgress = SCREWSLOADED;
            glutTimerFunc(50, processState, 0);
        }
    }
    else if (currentProgress == SCREWSLOADED)
    {
        currentProgress = GENERATING_WALLS;
        glutPostRedisplay();
        glutTimerFunc(100, generateWalls, 0);
    }
    else if (currentProgress == WALLSGENERATED)
    {
        // Start Animating Walls
        currentProgress = ANIMATING_WALLS;
        startSlicingAnimation();

        // Start Infill Generation in Background
        infillReady = false;
        infillError = false;
        // Need to ensure sliceInputs are stable (they are, collected before)
        // Spawn thread
        if (infillThread.joinable())
            infillThread.join(); // Safety cleanup
        infillThread = std::thread(backgroundInfillTask);
    }
    else if (currentProgress == WALLS_DONE)
    {
        // Walls finished. Wait for Infill if not ready.
        currentProgress = GENERATING_INFILL;
        glutPostRedisplay(); // Shows "Generating Infill..." if waiting
        checkInfillReady(0); // Polls flag
    }
    else if (currentProgress == INFILLGENERATED)
    {
        // Start Animating Infill
        currentProgress = ANIMATING_INFILL;
        startSlicingAnimation();
    }
    else if (currentProgress == REFINING_MESH)
    {
        // Refine mesh
        std::cout << "Refining mesh (max area 0.1)..." << std::endl;
        slicedMesh.refineMesh(1.0f);
        std::cout << "Refining done. New triangle count: " << slicedMesh.triangles.size() << std::endl;
        currentProgress = FULLYSLICED;
        glutPostRedisplay();
    }
}

// Renamed from startSlicing to indicate it just sets up params
void triggerSlicing()
{
    if (meshes.empty())
        return;
    // Just validation
    try
    {
        std::stof(sliceInputs[0]);
        std::stof(sliceInputs[1]);
        std::stoi(sliceInputs[2]);
        std::stof(sliceInputs[3]);

        // Params are valid
        sliceParamsCollected = true;

        // Trigger process
        glutTimerFunc(50, processState, 0);
    }
    catch (...)
    {
        std::cerr << "Invalid input for slicing." << std::endl;
        sliceParamsCollected = false;
    }
}

void recalculateScene()
{
    // Determine scene scaling based on the main model (or combined, but main is safer for stability)
    if (meshes.empty() || !meshes[0].active)
        return;

    // Use the main model's dimensions to set the scale factor
    globalMaxDimension = std::max(meshes[0].bounds.width(),
                                  std::max(meshes[0].bounds.height(), meshes[0].bounds.depth()));

    if (globalMaxDimension == 0)
        globalMaxDimension = 1.0f;
}

void addScrew(const std::string &filename)
{
    if (filename.empty())
        return;

    // Capture main mesh info before potentially resizing vector
    std::vector<Mesh::HoleLine> holes = meshes[0].holes;
    BoundingBox mainBounds = meshes[0].bounds;
    Vec3 mainCenter = meshes[0].center;
    Vec3 mainPos = meshes[0].positionOffset;

    if (holes.empty())
    {
        // Fallback: Add one screw to the side
        if (meshes.size() < 2)
            meshes.resize(2);

        // Disable any extra meshes if we had them before
        meshes.resize(2);

        if (meshes[1].loadFromSTL(filename.c_str()))
        {
            meshes[1].color = {1.0f, 0.0f, 0.0f}; // Red
            meshes[1].rotation = {0, 0, 0};

            // Calculate offset: Place it to the right of the first model
            float padding = mainBounds.width() * 0.1f;
            float xDist = (mainBounds.width() / 2.0f) + (meshes[1].bounds.width() / 2.0f) + padding;

            meshes[1].positionOffset = {xDist, 0.0f, 0.0f};
        }
    }
    else
    {
        // Add a screw for each hole
        size_t numHoles = holes.size();
        meshes.resize(1 + numHoles);

        // Load the model into the first screw slot
        if (!meshes[1].loadFromSTL(filename.c_str()))
        {
            // If failed, revert
            meshes.resize(1);
            return;
        }
        meshes[1].color = {1.0f, 0.0f, 0.0f}; // Red

        // Copy to other slots
        for (size_t i = 1; i < numHoles; ++i)
        {
            meshes[1 + i] = meshes[1];
        }

        // Position each screw
        for (size_t i = 0; i < numHoles; ++i)
        {
            auto &screw = meshes[1 + i];
            const auto &hole = holes[i];

            // Determine axis and position
            // Z-axis hole: top and bottom have same X, Y
            if (std::abs(hole.top.x - hole.bottom.x) < 0.001f &&
                std::abs(hole.top.y - hole.bottom.y) < 0.001f)
            {

                // Z-axis
                screw.positionOffset.x = hole.top.x - mainCenter.x + mainPos.x;
                screw.positionOffset.y = hole.top.y - mainCenter.y + mainPos.y;
                // Initial: screw bottom matches hole top (seat)
                screw.positionOffset.z = (hole.top.z - mainCenter.z + mainPos.z) - screw.bounds.minZ + screw.center.z;
                screw.rotation = {0, 0, 0};
            }
            else
            {
                // Y-axis
                screw.positionOffset.x = hole.top.x - mainCenter.x + mainPos.x;
                // Initial: screw bottom matches hole top (seat) - which is Y coordinate
                screw.positionOffset.y = (hole.top.y - mainCenter.y + mainPos.y) - screw.bounds.minZ + screw.center.z;
                screw.positionOffset.z = hole.top.z - mainCenter.z + mainPos.z;
                screw.rotation = {-90.0f, 0, 0};
            }
            std::cout << "Added screw " << i << " at " << screw.positionOffset.x << ", " << screw.positionOffset.y << ", " << screw.positionOffset.z << std::endl;
        }

        // Initialize Animation
        if (numHoles > 0)
        {
            bool wasAnimating = isAnimating;
            isAnimating = true;
            animProgress = 0.0f;

            // Resize and fill vectors
            // Mesh 0 is main. Meshes 1..N are screws.
            // Vector indices map to Mesh indices
            animStartPos.resize(meshes.size());
            animTargetPos.resize(meshes.size());
            gravityAnimStates.clear();
            gravityAnimStates.resize(meshes.size());

            // Initialize Main Mesh (no movement)
            animStartPos[0] = meshes[0].positionOffset;
            animTargetPos[0] = meshes[0].positionOffset;
            gravityAnimStates[0] = {0, {0,0,0}, 0.0f, 0, 0, false, true, 0, {0,0,0}};

            // Initialize Screws
            for (size_t i = 0; i < numHoles; ++i)
            {
                int meshIdx = 1 + i;
                auto &screw = meshes[meshIdx];
                const auto &hole = holes[i];

                animStartPos[meshIdx] = screw.positionOffset;

                // Calculate Target (Aligned Position only)
                Vec3 alignedPos;
                bool isZ = false;

                // Check axis again (duplicate logic, but safe)
                if (std::abs(hole.top.x - hole.bottom.x) < 0.001f &&
                    std::abs(hole.top.y - hole.bottom.y) < 0.001f)
                {
                    // Z-axis
                    isZ = true;
                }
                else
                {
                    // Y-axis
                    isZ = false;
                }

                alignedPos = meshes[0].getAlignedPosition(screw, hole.top, isZ);
                animTargetPos[meshIdx] = alignedPos;

                // Setup Gravity State
                GravityAnimationState state;
                state.meshIndex = meshIdx;
                state.active = true;
                state.finished = false;
                state.iteration = 0;
                state.maxIterations = 16; // Max attempts
                state.phase = 0; // Move first
                state.dir = isZ ? Vec3{0, 0, -1} : Vec3{0, -1, 0};
                state.step = isZ ? meshes[0].bounds.depth() : meshes[0].bounds.height();

                gravityAnimStates[meshIdx] = state;
            }

            // Start timer only if not already running
            if (!wasAnimating)
            {
                glutTimerFunc(16, updateAnimation, 0);
            }
        }
    }
}

void loadSecondModel()
{
    std::string filename = openFileDialog();
    addScrew(filename);
}

void loadSlicerGUI()
{
    // Check if all holes have screws
    // Logic: Total Meshes - 1 (Base) >= Number of Holes
    if (!meshes.empty() && meshes.size() > 0)
    {
        size_t numHoles = meshes[0].holes.size();
        size_t numScrews = meshes.size() - 1;

        if (numScrews >= numHoles)
        {
            isSliceSetupMode = true;
            currentSliceInputIndex = 0;
            currentInputBuffer = "";
            glutPostRedisplay();
        }
        else
        {
            std::cout << "Cannot start slice setup: Not all holes have screws." << std::endl;
        }
    }
}

// --- OpenGL Rendering ---

void initGL()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_BLEND); // For transparent UI
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat light0_pos[] = {10.0f, 10.0f, 10.0f, 0.0f};
    GLfloat light0_diffuse[] = {0.9f, 0.9f, 0.9f, 1.0f};
    GLfloat light0_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);

    GLfloat light1_pos[] = {-10.0f, -5.0f, 10.0f, 0.0f};
    GLfloat light1_diffuse[] = {0.4f, 0.4f, 0.4f, 1.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, light1_pos);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);

    GLfloat mat_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat mat_shininess[] = {60.0f};
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glClearColor(0.2f, 0.22f, 0.25f, 1.0f);
}

void pickObject(int x, int y)
{
    // 1. Clear color and depth buffers with black for ID picking
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // 2. Setup Camera (must match display)
    glTranslatef(0.0f, 0.0f, zoom);
    glRotatef(rotX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotY, 0.0f, 1.0f, 0.0f);
    float scale = 2.0f / globalMaxDimension;
    glScalef(scale, scale, scale);

    // 3. Disable lighting/dithering/blending for pure color picking
    glDisable(GL_LIGHTING);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);

    // 4. Render each mesh with a unique color ID
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        if (!meshes[i].active)
            continue;

        // ID = index + 1
        int id = (int)i + 1;
        GLubyte r = (id & 0x000000FF) >> 0;
        GLubyte g = (id & 0x0000FF00) >> 8;
        GLubyte b = (id & 0x00FF0000) >> 16;

        glColor3ub(r, g, b);
        meshes[i].draw(false); // Draw without internal color
    }

    // Render holes for Main mesh (index 0) for selection
    if (meshes.size() > 0 && meshes[0].active)
    {
        glLineWidth(5.0f); // Make them easier to hit

        const auto &holes = meshes[0].holes;
        for (size_t i = 0; i < holes.size(); ++i)
        {
            int id = (int)meshes.size() + 1 + i;
            GLubyte r = (id & 0x000000FF) >> 0;
            GLubyte g = (id & 0x0000FF00) >> 8;
            GLubyte b = (id & 0x00FF0000) >> 16;
            glColor3ub(r, g, b);

            const auto &hole = holes[i];
            glPushMatrix();
            // Replicate mesh transform
            glTranslatef(meshes[0].positionOffset.x, meshes[0].positionOffset.y, meshes[0].positionOffset.z);
            glRotatef(meshes[0].rotation.x, 1, 0, 0);
            glRotatef(meshes[0].rotation.y, 0, 1, 0);
            glRotatef(meshes[0].rotation.z, 0, 0, 1);
            glTranslatef(-meshes[0].center.x, -meshes[0].center.y, -meshes[0].center.z);

            glBegin(GL_LINES);
            glVertex3f(hole.top.x, hole.top.y, hole.top.z);
            glVertex3f(hole.bottom.x, hole.bottom.y, hole.bottom.z);
            glEnd();

            glPopMatrix();
        }
        glLineWidth(1.0f);
    }

    // 5. Read pixel at mouse position
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GLubyte pixel[3];
    // OpenGL 0,0 is bottom-left, window coordinates are top-left
    glReadPixels(x, viewport[3] - y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixel);

    // 6. Decode ID
    int id = pixel[0] + (pixel[1] << 8) + (pixel[2] << 16);

    // 7. Update selection
    if (id > 0 && id <= (int)meshes.size())
    {
        selectedMeshIndex = id - 1;
        std::cout << "Selected mesh: " << selectedMeshIndex << std::endl;
    }
    else if (id > (int)meshes.size())
    {
        int holeIndex = id - (int)meshes.size() - 1;
        std::cout << "Selected hole: " << holeIndex << std::endl;

        // Snap logic
        if (selectedMeshIndex > 0 && selectedMeshIndex < (int)meshes.size())
        {
            if (holeIndex >= 0 && holeIndex < (int)meshes[0].holes.size())
            {
                const auto &hole = meshes[0].holes[holeIndex];
                Vec3 mainCenter = meshes[0].center;
                Vec3 mainPos = meshes[0].positionOffset;

                // Snap horizontally (X, Y)
                meshes[selectedMeshIndex].positionOffset.x = hole.top.x - mainCenter.x + mainPos.x;
                meshes[selectedMeshIndex].positionOffset.y = hole.top.y - mainCenter.y + mainPos.y;

                std::cout << "Snapped mesh " << selectedMeshIndex << " to hole " << holeIndex << std::endl;
            }
        }
    }
    else
    {
        selectedMeshIndex = -1;
        std::cout << "Selection cleared." << std::endl;
    }

    // 8. Restore state
    glEnable(GL_LIGHTING);
    glEnable(GL_DITHER);
    glEnable(GL_BLEND);

    // Restore background color
    glClearColor(0.2f, 0.22f, 0.25f, 1.0f);
}

void display()
{
    glClearColor(0.2f, 0.22f, 0.25f, 1.0f); // Ensure bg color is correct
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Apply Camera
    glTranslatef(0.0f, 0.0f, zoom);
    glRotatef(rotX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotY, 0.0f, 1.0f, 0.0f);

    // Global Scaling
    float scale = 2.0f / globalMaxDimension;
    glScalef(scale, scale, scale);

    // Cache Matrices for Picking (before any other transforms)
    glGetDoublev(GL_MODELVIEW_MATRIX, cachedModelView);
    glGetDoublev(GL_PROJECTION_MATRIX, cachedProjection);
    glGetIntegerv(GL_VIEWPORT, cachedViewport);

    // Render Sliced Mesh and Infill being built
    if (sliceParamsCollected && slicedMesh.active)
    {
        float zLimit = slicedMesh.bounds.minZ + currentSliceLayer * currentLayerHeight;

        glPushMatrix();
        // Match Mesh Transform
        glTranslatef(slicedMesh.positionOffset.x, slicedMesh.positionOffset.y, slicedMesh.positionOffset.z);
        glRotatef(slicedMesh.rotation.x, 1, 0, 0);
        glRotatef(slicedMesh.rotation.y, 0, 1, 0);
        glRotatef(slicedMesh.rotation.z, 0, 0, 1);
        glTranslatef(-slicedMesh.center.x, -slicedMesh.center.y, -slicedMesh.center.z);

        // Draw Infill
        if (infillMesh.active)
        {
            // Clip Infill if animating it
            if (currentProgress == ANIMATING_INFILL && isSlicing)
            {
                GLdouble plane[] = {0, 0, -1, (double)zLimit};
                glClipPlane(GL_CLIP_PLANE0, plane);
                glEnable(GL_CLIP_PLANE0);
                infillMesh.draw(true, false);
                glDisable(GL_CLIP_PLANE0);
            }
            else if (currentProgress >= INFILLGENERATED)
            {
                // Draw full infill if done or waiting to start anim
                infillMesh.draw(true, false);
            }
        }

        // Draw Walls
        // Clip Walls if animating them
        if (currentProgress == ANIMATING_WALLS && isSlicing)
        {
            GLdouble plane[] = {0, 0, -1, (double)zLimit};
            glClipPlane(GL_CLIP_PLANE0, plane);
            glEnable(GL_CLIP_PLANE0);
            slicedMesh.draw(true, false);
            glDisable(GL_CLIP_PLANE0);
        }
        else
        {
            // Draw full walls if finished or not started yet (though logic prevents that)
            slicedMesh.draw(true, false);
        }

        glPopMatrix();
    }

    // Set depth function to LESS to prevent transparent ghost from drawing over opaque solid
    glDepthFunc(GL_LESS);

    // Render all meshes (Originals + Screws)
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        bool isGhost = (i == 0 && (isSlicing || sliceParamsCollected));

        if (isGhost)
        {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f); // Push fragments back
        }

        meshes[i].draw(true); // Draw with internal color

        if (isGhost)
        {
            glDisable(GL_POLYGON_OFFSET_FILL);
        }

        if ((int)i == selectedMeshIndex)
        {
            meshes[i].drawBoundingBox();
        }
    }

    if (!sliceParamsCollected)
    {
        loadBtn.draw(windowW, windowH);
        sliceBtn.draw(windowW, windowH);
    }
    else if (currentProgress == FULLYSLICED || currentProgress == PAINTING)
    {
        paintBtn.draw(windowW, windowH);
        forceBtn.draw(windowW, windowH);
        if (!displayedForceStr.empty() && displayedForceStr != "Invalid Force")
        {
            simBtn.draw(windowW, windowH);
        }
    }

    // Switch to 2D for Overlay
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowW, windowH, 0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    if (isSliceSetupMode || isForceSetupMode)
    {
        // Semi-transparent overlay
        glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
        glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(windowW, 0);
        glVertex2i(windowW, windowH);
        glVertex2i(0, windowH);
        glEnd();

        // Input Box - Move to Bottom Left as requested?
        int bw = 300;
        int bh = 100;
        int cx = 20 + bw / 2;            // Left margin 20
        int cy = windowH - 100 + bh / 2; // Bottom area

        glColor4f(0.2f, 0.2f, 0.2f, 0.9f);
        glBegin(GL_QUADS);
        glVertex2i(cx - bw / 2, cy - bh / 2);
        glVertex2i(cx + bw / 2, cy - bh / 2);
        glVertex2i(cx + bw / 2, cy + bh / 2);
        glVertex2i(cx - bw / 2, cy + bh / 2);
        glEnd();

        glColor3f(1.0f, 1.0f, 1.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        glVertex2i(cx - bw / 2, cy - bh / 2);
        glVertex2i(cx + bw / 2, cy - bh / 2);
        glVertex2i(cx + bw / 2, cy + bh / 2);
        glVertex2i(cx - bw / 2, cy + bh / 2);
        glEnd();

        if (isSliceSetupMode)
        {
            // Prompt
            glColor3f(1.0f, 1.0f, 1.0f);
            const char *prompt = slicePrompts[currentSliceInputIndex];
            renderText(cx - 140, cy - 10, prompt);

            // Input Value
            std::string displayVal = currentInputBuffer + "_";
            renderText(cx - 140, cy + 20, displayVal.c_str());
        }
        else if (isForceSetupMode)
        {
            // Prompt
            glColor3f(1.0f, 1.0f, 1.0f);
            renderText(cx - 140, cy - 10, "Enter Force (e.g. 100N, 1.5kN):");

            // Input Value
            std::string displayVal = forceInputBuffer + "_";
            renderText(cx - 140, cy + 20, displayVal.c_str());
        }
    }

    if (sliceParamsCollected)
    {
        // Draw Parameters at Bottom Left
        glColor3f(1.0f, 1.0f, 1.0f);
        int startY = windowH - 100;
        int startX = 20;
        int lineHeight = 20;

        char buf[256];
        snprintf(buf, 256, "Nozzle: %s mm", sliceInputs[0].c_str());
        renderText(startX, startY, buf);

        snprintf(buf, 256, "Layer: %s mm", sliceInputs[1].c_str());
        renderText(startX, startY + lineHeight, buf);

        snprintf(buf, 256, "Walls: %s", sliceInputs[2].c_str());
        renderText(startX, startY + lineHeight * 2, buf);

        snprintf(buf, 256, "Infill: %s %%", sliceInputs[3].c_str());
        renderText(startX, startY + lineHeight * 3, buf);
    }

    if (!displayedForceStr.empty())
    {
        glColor3f(1.0f, 1.0f, 0.0f); // Yellow
        renderText(windowW - 200, windowH - 50, displayedForceStr.c_str());
    }
    if (isSimulating) {
        // Draw Progress Bar Box
        int barW = 400;
        int barH = 30;
        int cx = windowW / 2;
        int cy = windowH / 2;
        
        // Background
        glColor4f(0.2f, 0.2f, 0.2f, 0.9f);
        glBegin(GL_QUADS);
            glVertex2i(cx - barW/2, cy - barH/2);
            glVertex2i(cx + barW/2, cy - barH/2);
            glVertex2i(cx + barW/2, cy + barH/2);
            glVertex2i(cx - barW/2, cy + barH/2);
        glEnd();
        
        // Border
        glColor3f(1.0f, 1.0f, 1.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
            glVertex2i(cx - barW/2, cy - barH/2);
            glVertex2i(cx + barW/2, cy - barH/2);
            glVertex2i(cx + barW/2, cy + barH/2);
            glVertex2i(cx - barW/2, cy + barH/2);
        glEnd();
        
        // Fill (Green)
        float pct = simProgress; 
        if (pct < 0) pct = 0; if (pct > 1) pct = 1;
        int fillW = (int)(barW * pct);
        
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUADS);
            glVertex2i(cx - barW/2 + 2, cy - barH/2 + 2);
            glVertex2i(cx - barW/2 + fillW - 2, cy - barH/2 + 2);
            glVertex2i(cx - barW/2 + fillW - 2, cy + barH/2 - 2);
            glVertex2i(cx - barW/2 + 2, cy + barH/2 - 2);
        glEnd();
        
        // Text
        glColor3f(1.0f, 1.0f, 1.0f);
        char buf[64];
        snprintf(buf, 64, "Simulating... %.0f%%", pct * 100.0f);
        int textW = getTextWidth(buf);
        renderText(cx - textW / 2, cy - 5, buf);
    }

    // Draw Status Messages at Top Middle
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow
    char buf[256];

    if (currentProgress == SCREWSLOADED && sliceParamsCollected)
    {
        snprintf(buf, 256, "Preparing to slice...");
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (currentProgress == GENERATING_WALLS)
    {
        snprintf(buf, 256, "Generating Walls...");
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (currentProgress == GENERATING_INFILL && sliceParamsCollected)
    {
        snprintf(buf, 256, "Generating Infill...");
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (isSlicing)
    {
        if (currentProgress == ANIMATING_WALLS)
        {
            snprintf(buf, 256, "Building Walls: %d / %d", currentSliceLayer, totalLayers);
        }
        else if (currentProgress == ANIMATING_INFILL)
        {
            snprintf(buf, 256, "Building Infill: %d / %d", currentSliceLayer, totalLayers);
        }
        else
        {
            snprintf(buf, 256, "Building Layer: %d / %d", currentSliceLayer, totalLayers);
        }
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (currentProgress == REFINING_MESH)
    {
        snprintf(buf, 256, "Refining Mesh...");
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (currentProgress == FULLYSLICED)
    {
        snprintf(buf, 256, "Slicing Complete");
        renderText(windowW / 2 - 100, 40, buf);
    }
    else if (currentProgress == PAINTING)
    {
        snprintf(buf, 256, "Painting Mode");
        renderText(windowW / 2 - 100, 40, buf);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    windowW = w;
    windowH = h;
    if (h == 0)
        h = 1;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)w / (float)h, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

// --- Input Handling ---

void keyboard(unsigned char key, int x, int y)
{
    // Ensure thread safety on reset or exit
    if (key == 'r' || key == 'R' || key == 27)
    {
        if (infillThread.joinable())
            infillThread.join();
        // Assuming 'r' resets app, but we don't have reset logic visible here.
        // If 'r' was supported, we'd need to handle it. For now, just safety check.
    }

    if (isSliceSetupMode)
    {
        if (key == 13)
        { // Enter
            if (!currentInputBuffer.empty())
            {
                sliceInputs[currentSliceInputIndex] = currentInputBuffer;
                currentInputBuffer = "";
                currentSliceInputIndex++;
                if (currentSliceInputIndex >= 4)
                {
                    isSliceSetupMode = false;
                    sliceParamsCollected = true;
                    triggerSlicing(); // Trigger slicing flow
                }
            }
        }
        else if (key == 8 || key == 127)
        { // Backspace
            if (!currentInputBuffer.empty())
            {
                currentInputBuffer.pop_back();
            }
        }
        else if ((key >= '0' && key <= '9') || key == '.')
        {
            currentInputBuffer += key;
        }
        glutPostRedisplay();
        return;
    }

    if (isForceSetupMode)
    {
        if (key == 13)
        { // Enter
            parseForceInput();
            isForceSetupMode = false;
        }
        else if (key == 8 || key == 127)
        { // Backspace
            if (!forceInputBuffer.empty())
            {
                forceInputBuffer.pop_back();
            }
        }
        else
        {
            // Allow digits, '.', 'k', 'N', 'n', 'K'
            if (isdigit(key) || key == '.' || key == 'k' || key == 'K' || key == 'n' || key == 'N')
            {
                forceInputBuffer += key;
            }
        }
        glutPostRedisplay();
        return;
    }

    if (tolower(key) == tolower(loadBtn.shortcut) && !sliceParamsCollected)
    {
        loadBtn.onClick();
        glutPostRedisplay();
    }
    else if (tolower(key) == tolower(sliceBtn.shortcut) && !sliceParamsCollected)
    {
        sliceBtn.onClick();
        glutPostRedisplay();
    }
    else if (tolower(key) == tolower(paintBtn.shortcut) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
    {
        paintBtn.onClick();
        glutPostRedisplay();
    }
    else if (tolower(key) == tolower(forceBtn.shortcut) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
    {
        forceBtn.onClick();
        glutPostRedisplay();
    }
    else if (tolower(key) == tolower(simBtn.shortcut) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
    {
        if (!displayedForceStr.empty() && displayedForceStr != "Invalid Force")
        {
            simBtn.onClick();
            glutPostRedisplay();
        }
    }
    else if (key == 27)
    { // Escape
        if (isSliceSetupMode)
        {
            isSliceSetupMode = false; // Cancel
            currentSliceInputIndex = 0;
            currentInputBuffer = "";
            glutPostRedisplay();
        }
        else if (isForceSetupMode)
        {
            isForceSetupMode = false;
            forceInputBuffer = "";
            glutPostRedisplay();
        }
        else
        {
            // Already joined above if needed, but good for explicit exit path
            exit(0);
        }
    }
}

void mouse(int button, int state, int x, int y)
{
    lastMouseX = x;
    lastMouseY = y;

    // Block interaction if in Modal Mode (Force Setup or Slice Setup)
    // Assuming clicking outside might close it? Or just consume the click?
    // Let's consume it to prevent accidental clicks on buttons/mesh.
    if (isForceSetupMode)
    {
        // Optional: click outside input box to close?
        // For now, just consume.
        return;
    }

    // Check UI Click
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        if (loadBtn.isInside(x, y) && !sliceParamsCollected)
        {
            loadBtn.onClick();
            glutPostRedisplay();
            return;
        }
        else if (sliceBtn.isInside(x, y) && !sliceParamsCollected)
        {
            sliceBtn.onClick();
            glutPostRedisplay();
            return;
        }
        else if (paintBtn.isInside(x, y) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
        {
            paintBtn.onClick();
            glutPostRedisplay();
            return;
        }
        else if (forceBtn.isInside(x, y) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
        {
            forceBtn.onClick();
            glutPostRedisplay();
            return;
        }
        else if (simBtn.isInside(x, y) && (currentProgress == FULLYSLICED || currentProgress == PAINTING))
        {
            // Only allow if force is set
            if (!displayedForceStr.empty() && displayedForceStr != "Invalid Force")
            {
                simBtn.onClick();
                glutPostRedisplay();
                return;
            }
        }

        if (currentProgress == PAINTING)
        {
            paintOnSurface(x, y);
            glutPostRedisplay();
            // Don't rotate, but allow drag (state 1)
            mouseState = 1;
            return;
        }

        // Pick Object
        pickObject(x, y);
        glutPostRedisplay();
    }

    if (button == GLUT_LEFT_BUTTON)
    {
        mouseState = (state == GLUT_DOWN) ? 1 : 0;
    }
    else if (button == GLUT_RIGHT_BUTTON)
    {
        mouseState = (state == GLUT_DOWN) ? 2 : 0;
    }
    else if (button == 3)
    { // Scroll Up
        zoom += 0.2f;
        glutPostRedisplay();
    }
    else if (button == 4)
    { // Scroll Down
        zoom -= 0.2f;
        glutPostRedisplay();
    }
}

void motion(int x, int y)
{
    // Check UI Hover
    bool prevHover = loadBtn.hover;
    loadBtn.hover = loadBtn.isInside(x, y);
    if (prevHover != loadBtn.hover)
        glutPostRedisplay();
    bool prevSliceHover = sliceBtn.hover;
    sliceBtn.hover = sliceBtn.isInside(x, y);
    if (prevSliceHover != sliceBtn.hover)
        glutPostRedisplay();

    bool prevPaintHover = paintBtn.hover;
    paintBtn.hover = paintBtn.isInside(x, y);
    if (prevPaintHover != paintBtn.hover)
        glutPostRedisplay();

    bool prevForceHover = forceBtn.hover;
    forceBtn.hover = forceBtn.isInside(x, y);
    if (prevForceHover != forceBtn.hover)
        glutPostRedisplay();

    bool prevSimHover = simBtn.hover;
    simBtn.hover = simBtn.isInside(x, y);
    if (prevSimHover != simBtn.hover)
        glutPostRedisplay();

    // Painting Logic (Drag)
    if (currentProgress == PAINTING && mouseState == 1)
    { // Left Button Drag
        paintOnSurface(x, y);
        lastMouseX = x;
        lastMouseY = y;
        return; // Skip camera rotation
    }

    // Camera Logic
    int dx = x - lastMouseX;
    int dy = y - lastMouseY;

    // Only move camera if we aren't clicking the button (simple check, state handles drags)
    if (mouseState == 1)
    { // Rotate
        rotY += dx * 0.5f;
        rotX += dy * 0.5f;
        glutPostRedisplay();
    }
    else if (mouseState == 2)
    { // Zoom
        zoom += dy * 0.05f;
        glutPostRedisplay();
    }

    lastMouseX = x;
    lastMouseY = y;
}

void passiveMotion(int x, int y)
{
    bool prevHover = loadBtn.hover;
    loadBtn.hover = loadBtn.isInside(x, y);
    if (prevHover != loadBtn.hover)
        glutPostRedisplay();
    bool prevSliceHover = sliceBtn.hover;
    sliceBtn.hover = sliceBtn.isInside(x, y);
    if (prevSliceHover != sliceBtn.hover)
        glutPostRedisplay();
    bool prevPaintHover = paintBtn.hover;
    paintBtn.hover = paintBtn.isInside(x, y);
    if (prevPaintHover != paintBtn.hover)
        glutPostRedisplay();

    bool prevForceHover = forceBtn.hover;
    forceBtn.hover = forceBtn.isInside(x, y);
    if (prevForceHover != forceBtn.hover)
        glutPostRedisplay();
}

void checkSimulation(int value) {
    if (simProgress < 1.0f && simStatusText != "Failed!") {
        // Still running, check again in 100ms
        glutTimerFunc(100, checkSimulation, 0);
        glutPostRedisplay(); // Update the progress bar
    } else {
        // Finished!
        if (simThread.joinable()) simThread.join();
        
        if (simStatusText != "Failed!") {
            slicedMesh = simResultMesh; // Apply result to main mesh
            std::cout << "Simulation Complete. Max Stress/Disp: " << simMaxS << std::endl;
        }
        
        isSimulating = false; // Unblock UI
        glutPostRedisplay();
    }
}

void simulationWorker(Mesh inputMesh, Material mat, float force, Vec3 dir, Vec3 paintColor) {
    FeaSolver solver;
    
    // 1. Build (Slow)
    solver.buildSystem(inputMesh, mat);
    
    // 2. Load
    solver.applyLoadFromPaint(inputMesh, paintColor, force, dir);
    solver.autoFixGround(0.001f);
    
    // 3. Solve (Slow) - Passes atomic progress pointer
    if (solver.solve(&simProgress)) {
        // 4. Compute Colors & Deform (Fast)
        solver.computeStressColorMap(inputMesh, simMinS, simMaxS);
        solver.applyDeformation(inputMesh, 1.0f); // 1.0x Scale
        
        // Write to global result buffer
        simResultMesh = inputMesh;
        simStatusText = "Done!";
    } else {
        simStatusText = "Failed!";
    }
}



// --- Main ---

int main(int argc, char **argv)
{
    std::string filename;

    if (argc < 2)
    {
        std::cout << "No file provided. Opening file dialog..." << std::endl;
        filename = openFileDialog();
        if (filename.empty())
        {
            std::cout << "No file selected. Exiting." << std::endl;
            return 0;
        }
    }
    else
    {
        filename = argv[1];
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(800, 600);
    glutCreateWindow("STL Viewer with UI");

    // Initialize UI
    loadBtn = {20, 20, 160, 40, "Load screw (A)", false, 'a', loadSecondModel};
    sliceBtn = {200, 20, 160, 40, "Slice Setup (S)", false, 's', loadSlicerGUI};
    paintBtn = {20, 20, 160, 40, "Paint (P)", false, 'p', enterPaintMode};
    forceBtn = {200, 20, 160, 40, "Enter force (F)", false, 'f', enterForceMode};
    simBtn = {380, 20, 160, 40, "Simulate (R)", false, 'r', startSimulation};
    // Initialize meshes vector
    meshes.resize(1);

    // Load the first file
    if (!meshes[0].loadFromSTL(filename.c_str()))
    {
        return 1;
    }
    currentProgress = MODELLOADED; // Update State

    meshes[0].color = {0.3f, 0.6f, 0.9f}; // Original Blue
    meshes[0].positionOffset = {0, 0, 0};

    // Detect hole for screw
    meshes[0].detectHole();

    recalculateScene();

    // Load extra screws from CLI
    // Check if the last 4 arguments are slicing parameters
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
    {
        args.push_back(argv[i]);
    }

    // We expect at least: prog base [screw...] nozzle layer wall infill
    // So minimum argc = 1 (prog) + 1 (base) + 4 (params) = 6
    bool hasSliceParams = false;
    int screwEndIndex = argc;

    if (argc >= 6)
    {
        // Simple check: see if last 4 are numeric
        try
        {
            float p1 = std::stof(args[argc - 4]);
            float p2 = std::stof(args[argc - 3]);
            int p3 = std::stoi(args[argc - 2]);
            float p4 = std::stof(args[argc - 1]);
            (void)p1;
            (void)p2;
            (void)p3;
            (void)p4; // suppress unused warning

            hasSliceParams = true;
            screwEndIndex = argc - 4;

            // Populate slice inputs
            sliceInputs[0] = args[argc - 4];
            sliceInputs[1] = args[argc - 3];
            sliceInputs[2] = args[argc - 2];
            sliceInputs[3] = args[argc - 1];
        }
        catch (...)
        {
            // Not slicing params
            hasSliceParams = false;
        }
    }

    for (int i = 2; i < screwEndIndex; ++i)
    {
        addScrew(argv[i]);
    }

    // Auto-start slicing if params were provided
    if (hasSliceParams)
    {
        sliceParamsCollected = true;
        autoSlice = true;
        // We do NOT call triggerSlicing() here.
        // We let updateAnimation (if running) or a startup timer trigger it.
        // If animation is running, it will call processState on finish.
        // If animation is NOT running (no screws), we need to kickstart it.
        if (!isAnimating)
        {
            // Treat as SCREWSLOADED effectively
            // glutTimerFunc(100, processState, 0);
            // But we need to ensure main loop is running.
            // Timer func works before main loop.
            glutTimerFunc(500, processState, 0); // Delay slightly to allow window to appear
        }
    }

    initGL();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion); // For hover effect

    std::cout << "Viewer started." << std::endl;
    std::cout << "Controls: Mouse to Rotate/Zoom." << std::endl;
    std::cout << "Press 'A' or click the button to load a second model." << std::endl;
    std::cout << "Click on an object to select it (Bounding Box)." << std::endl;

    glutMainLoop();
    return 0;
}

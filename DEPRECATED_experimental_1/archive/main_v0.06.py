import open3d as o3d
import sys
import os
import subprocess
import numpy as np
import copy
import math

# Global list to track loaded meshes
MESHES = []
# Global list to store detected hole centers (x, y, z)
HOLE_LOCATIONS = []
# Counter to track which hole to fill next
SCREW_COUNT = 0

# Only import tkinter if NOT on macOS to avoid NSApplication crashes
if sys.platform != 'darwin':
    import tkinter as tk
    from tkinter import filedialog

def get_file_path():
    """
    Opens a file picker dialog.
    On macOS: Uses AppleScript (osascript) to avoid Tkinter/Open3D conflicts.
    On Windows/Linux: Uses Tkinter.
    """
    if sys.platform == 'darwin':
        # Native macOS file picker via AppleScript
        try:
            script = """
            set theFile to choose file with prompt "Select an STL file" of type {"stl"}
            POSIX path of theFile
            """
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None # User cancelled
        except Exception as e:
            print(f"Error opening macOS file picker: {e}")
            return None
    else:
        # Standard Tkinter picker for Windows/Linux
        try:
            root = tk.Tk()
            root.withdraw() 
            root.update()
            file_path = filedialog.askopenfilename(
                title="Select an STL file",
                filetypes=[("STL Files", "*.stl"), ("All Files", "*.*")]
            )
            root.update()
            root.destroy()
            return file_path
        except Exception:
            return None

def load_mesh(file_path):
    """
    Loads an STL file and computes normals.
    Returns None if loading fails.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_triangles():
            print(f"Error: '{os.path.basename(file_path)}' is empty or invalid.")
            return None
            
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

def fit_circle_least_squares(points_2d):
    """
    Fits a circle to a set of 2D points using linear least squares.
    Returns center (xc, yc) and radius r.
    Solving: x^2 + y^2 + Dx + Ey + F = 0
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    # Ax = B
    # A = [x, y, 1]
    # x_coeffs = [-D, -E, -F]
    # B = x^2 + y^2
    
    A = np.column_stack((x, y, np.ones(len(x))))
    B = x**2 + y**2
    
    # Use lstsq to solve
    try:
        solution, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        D, E, F = solution
        
        xc = D / 2.0
        yc = E / 2.0
        # r = np.sqrt(xc**2 + yc**2 + F) 
        
        return np.array([xc, yc])
    except Exception as e:
        print(f"     -> Circle fit failed: {e}")
        return np.mean(points_2d, axis=0) # Fallback to centroid

def detect_hole_locations(mesh):
    """
    Analyzes the base mesh to find vertical holes on the top surface.
    IMPROVED METHOD: 
    1. Finds vertical walls.
    2. Uses Least Squares Circle Fit to find the PRECISE center (avoiding overlap).
    3. Scans inside for 'shelf' (counterbore floor).
    """
    print("Scanning base mesh for holes...")
    
    # 1. Get Geometry Data
    aabb = mesh.get_axis_aligned_bounding_box()
    max_z = aabb.get_max_bound()[2]
    
    pts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # --- PRE-CALCULATE SHELF CANDIDATES ---
    is_shelf = (pts[:, 2] < (max_z - 1.0)) & (normals[:, 2] > 0.9)
    shelf_pts = pts[is_shelf]
    
    # --- FIND VERTICAL WALLS (for XY detection) ---
    z_threshold = max_z - 25.0 
    
    mask = (pts[:, 2] > z_threshold) & (np.abs(normals[:, 2]) < 0.2)
    wall_pts = pts[mask]
    
    if len(wall_pts) == 0:
        print("  -> No vertical wall geometry found in top section.")
        return [aabb.get_center()]
        
    # Project to XY plane for clustering
    pts_2d = wall_pts[:, :2]
    
    # Cluster using DBSCAN
    pcd_2d = o3d.geometry.PointCloud()
    pts_3d_for_clustering = np.zeros((len(pts_2d), 3))
    pts_3d_for_clustering[:, :2] = pts_2d
    pcd_2d.points = o3d.utility.Vector3dVector(pts_3d_for_clustering)
    
    labels = np.array(pcd_2d.cluster_dbscan(eps=2.5, min_points=10, print_progress=False))
    
    if len(labels) == 0:
        return [aabb.get_center()]

    max_label = labels.max()
    print(f"  -> Detected {max_label + 1} potential clusters (Vertical Walls).")
    
    candidates = []
    
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pts = pts_2d[cluster_indices]
        
        # Calculate Dimensions
        min_b = np.min(cluster_pts, axis=0)
        max_b = np.max(cluster_pts, axis=0)
        dims = max_b - min_b
        diagonal = np.linalg.norm(dims)
        
        # --- NEW: Precise Circle Fitting ---
        # Instead of simple mean, we fit a circle to the wall points.
        # This centers the screw perfectly in the hole, maximizing clearance.
        if len(cluster_pts) > 3:
            center_xy = fit_circle_least_squares(cluster_pts)
        else:
            center_xy = np.mean(cluster_pts, axis=0)

        # --- DETECT Z-DEPTH (Counterbore Shelf) ---
        target_z = max_z 
        
        if len(shelf_pts) > 0:
            dx = shelf_pts[:, 0] - center_xy[0]
            dy = shelf_pts[:, 1] - center_xy[1]
            dist_sq = dx*dx + dy*dy
            
            nearby_shelf_indices = np.where(dist_sq < 144.0)[0] # 12mm radius
            
            if len(nearby_shelf_indices) > 0:
                nearby_z = shelf_pts[nearby_shelf_indices, 2]
                target_z = np.median(nearby_z)
                print(f"     Feature {i}: Found internal shelf at Z={target_z:.2f}")
        
        center_3d = np.array([center_xy[0], center_xy[1], target_z])
        
        # FILTERING LOGIC:
        if diagonal > 8.0 and diagonal < 45.0:
            candidates.append(center_3d)
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> CANDIDATE (Precise Center)")
        else:
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> Ignored (Size)")
            
    # Sort candidates
    candidates.sort(key=lambda p: (p[0], p[1]))
    
    # Merge close duplicates
    final_candidates = []
    if len(candidates) > 0:
        curr = candidates[0]
        group = [curr]
        
        for next_cand in candidates[1:]:
            dist = np.linalg.norm(next_cand - curr)
            if dist < 6.0: 
                group.append(next_cand)
            else:
                avg_center = np.mean(group, axis=0)
                final_candidates.append(avg_center)
                curr = next_cand
                group = [curr]
        final_candidates.append(np.mean(group, axis=0))
    
    print(f"  -> Identified {len(final_candidates)} distinct hole locations.")
    
    if len(final_candidates) == 0:
         return [aabb.get_center()]
         
    return final_candidates

def find_screw_shoulder_z(mesh):
    """
    Analyzes the screw mesh (assuming it's vertical, head-up) to find the Z-coordinate 
    where the head ends and the shaft begins (the shoulder).
    """
    pts = np.asarray(mesh.vertices)
    if len(pts) == 0: return 0
    
    z = pts[:, 2]
    min_z, max_z = np.min(z), np.max(z)
    height = max_z - min_z
    
    # We slice the mesh from Top (Head) down to Bottom (Shaft)
    num_slices = 100
    step = height / num_slices
    
    # 1. Determine Head Diameter from the top 5% of the mesh
    top_threshold = max_z - (height * 0.05)
    top_mask = (z > top_threshold)
    
    if not np.any(top_mask): 
        return max_z
        
    top_pts = pts[top_mask]
    top_width = np.max(top_pts[:,0]) - np.min(top_pts[:,0])
    top_depth = np.max(top_pts[:,1]) - np.min(top_pts[:,1])
    head_diameter = (top_width + top_depth) / 2.0
    
    # 2. Scan downwards to find where diameter drops significantly
    for i in range(num_slices):
        z_high = max_z - (i * step)
        z_low = z_high - step
        
        mask = (z >= z_low) & (z < z_high)
        if not np.any(mask): continue
        
        slice_pts = pts[mask]
        w = np.max(slice_pts[:,0]) - np.min(slice_pts[:,0])
        d = np.max(slice_pts[:,1]) - np.min(slice_pts[:,1])
        slice_dia = (w + d) / 2.0
        
        # If diameter drops below 85% of head diameter, we found the shoulder
        if slice_dia < (head_diameter * 0.85):
            print(f"  -> Screw Shoulder detected at Z={z_high:.2f}")
            return z_high

    return min_z 

def fit_screw_to_hole(screw, base, hole_index):
    """
    Fits a screw into the specific hole index detected on the base.
    """
    global HOLE_LOCATIONS
    
    print(f"Fitting Screw #{hole_index + 1}...")
    
    # 1. Determine Target Location (XY)
    if len(HOLE_LOCATIONS) == 0:
        # Fallback to center if detection failed
        target_pos = base.get_axis_aligned_bounding_box().get_center()
        print("  -> No holes detected, using base center.")
    else:
        # Wrap around if we have more screws than holes
        target_idx = hole_index % len(HOLE_LOCATIONS)
        target_pos = HOLE_LOCATIONS[target_idx]
        print(f"  -> Targeting Hole {target_idx + 1} at {target_pos[:2]}")

    # 2. XY Alignment
    screw_center = screw.get_axis_aligned_bounding_box().get_center()
    dx = target_pos[0] - screw_center[0]
    dy = target_pos[1] - screw_center[1]
    
    screw.translate([dx, dy, 0])

    # 3. Z Alignment (Shoulder to Top)
    shoulder_z = find_screw_shoulder_z(screw)
    base_shelf_z = target_pos[2] 
    
    dz = base_shelf_z - shoulder_z
    screw.translate([0, 0, dz])
    
    print("  -> Fit complete.")

def add_file_callback(vis):
    """
    Callback function triggered by pressing 'A'.
    Pauses the visualizer to open a file dialog and adds the new mesh.
    """
    global SCREW_COUNT
    print("\n--- Add File Triggered ---")
    
    file_path = get_file_path()
    
    if not file_path:
        print("No file selected.")
        return False
        
    print(f"Loading {file_path}...")
    mesh = load_mesh(file_path)
    
    if mesh:
        # If this is a Screw (Mesh > 0)
        if len(MESHES) > 0:
            print("Detected additional mesh (Screw).")
            base_mesh = MESHES[0]
            
            # Color screws differently
            mesh.paint_uniform_color([0.8, 0.2, 0.2]) # Red
            
            # Perform the fit
            fit_screw_to_hole(mesh, base_mesh, SCREW_COUNT)
            SCREW_COUNT += 1

        # Store in global list
        MESHES.append(mesh)

        # Add to scene
        vis.add_geometry(mesh, reset_bounding_box=True)
        vis.poll_events()
        vis.update_renderer()
        
        print(f"Successfully added: {os.path.basename(file_path)}")
        return True
    
    return False

# ==========================================
#  CARVING / SIMULATION LOGIC
# ==========================================

def get_input_default(prompt, default_val, cast_type=float):
    """Helper to get input with a default value."""
    try:
        val = input(f"{prompt} [{default_val}]: ").strip()
        if not val:
            return default_val
        return cast_type(val)
    except Exception:
        return default_val

def generate_gyroid(x, y, z, scale):
    """Returns Gyroid values for a grid of points."""
    return (np.sin(x * scale) * np.cos(y * scale) +
            np.sin(y * scale) * np.cos(z * scale) +
            np.sin(z * scale) * np.cos(x * scale))

def generate_honeycomb(x, y, z, scale):
    """
    Simple hex-like grid approximation using sine waves.
    Not a perfect honeycomb but computationally fast for simulation.
    """
    # 2D honeycomb pattern extended in Z
    return (np.sin(x*scale) + np.sin((x*0.5 + y*0.866)*scale) + 
            np.sin((x*0.5 - y*0.866)*scale))

def perform_carving_tensor(mesh, layer_h, nozzle_w, walls, infill_pct, infill_type):
    """
    Uses Open3D Tensor API (RaycastingScene) to compute the carved geometry.
    Returns a VoxelGrid (or PointCloud) representing the printed part.
    """
    print("\n--- Processing Geometry (this may take a moment) ---")
    
    # Check for Tensor API availability
    try:
        import open3d.core as o3c
        from open3d.t.geometry import RaycastingScene, TriangleMesh
    except ImportError:
        print("Error: Open3D Tensor API not available. Update Open3D (pip install open3d --upgrade).")
        return None

    # 1. Setup Raycasting Scene
    # Convert legacy mesh to tensor mesh
    t_mesh = TriangleMesh.from_legacy(mesh)
    scene = RaycastingScene()
    scene.add_triangles(t_mesh)
    
    # 2. Define Query Grid
    # We use 'nozzle_width' as the grid resolution for X/Y. 
    # For Z, we could use layer_height, but cubic voxels look better.
    # Let's settle on a cubic resolution = nozzle_width (or slightly coarser for speed).
    
    res = nozzle_w
    # Limit resolution for very large objects to prevent RAM crash
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()
    dims = max_b - min_b
    
    # If roughly > 300 voxels in any dim, clamp resolution
    max_voxels = 250
    if np.max(dims) / res > max_voxels:
        res = np.max(dims) / max_voxels
        print(f"  -> Large model detected. Adjusting simulation resolution to {res:.2f}mm")

    # Generate grid coordinates
    xs = np.arange(min_b[0], max_b[0], res)
    ys = np.arange(min_b[1], max_b[1], res)
    zs = np.arange(min_b[2], max_b[2], res) # or layer_h
    
    # Meshgrid is heavy. Create query tensor directly.
    # Creating dense grid:
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    query_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1).astype(np.float32)
    
    print(f"  -> Simulating {len(query_points)} points...")
    
    query_tensor = o3c.Tensor(query_points, dtype=o3c.float32)

    # 3. Compute Signed Distance (SDF)
    # Negative = Inside, Positive = Outside
    signed_dist = scene.compute_signed_distance(query_tensor)
    sdf_vals = signed_dist.numpy()

    # 4. Define Regions
    wall_thickness = walls * nozzle_w
    
    # Shell: Inside the object, but close to surface
    # SDF is negative inside. So shell is where SDF is between 0 and -wall_thickness
    is_inside_mesh = sdf_vals <= 0
    is_shell = (sdf_vals <= 0) & (sdf_vals > -wall_thickness)
    is_core = sdf_vals <= -wall_thickness
    
    # 5. Calculate Infill
    # Scale factor for infill density
    # Base period ~ 10mm. 
    scale_factor = (infill_pct / 20.0) * (2.0 * np.pi / 10.0)
    
    if infill_type == 'gyroid':
        pattern_vals = generate_gyroid(query_points[:,0], query_points[:,1], query_points[:,2], scale_factor)
        # Gyroid > 0 is roughly 50%. Adjust threshold for percentage? 
        # For simple visual, > 0 is fine.
        keep_infill = pattern_vals > 0
    elif infill_type == 'honeycomb':
        pattern_vals = generate_honeycomb(query_points[:,0], query_points[:,1], query_points[:,2], scale_factor)
        keep_infill = pattern_vals > 0
    elif infill_type == 'tri-hexagon':
        # Simple triangular grid interference
        v1 = np.sin(query_points[:,0]*scale_factor)
        v2 = np.sin((query_points[:,0]*0.5 + query_points[:,1]*0.866)*scale_factor)
        keep_infill = (v1 * v2) > 0
    else: # triangles / lines
        # Simple grid lines
        v1 = np.sin(query_points[:,0]*scale_factor)
        v2 = np.sin(query_points[:,1]*scale_factor)
        keep_infill = (v1 + v2) > 0.5
        
    # Combine Logic
    # Final Voxel = (Shell) OR (Core AND Infill_Pattern)
    final_mask = is_shell | (is_core & keep_infill)
    
    # Filter points
    valid_points = query_points[final_mask]
    
    # Colorize: Shell = Grey, Infill = Orange
    colors = np.zeros_like(valid_points)
    
    # Re-evaluate mask on reduced set to assign colors? 
    # Easier: assign colors based on mask index before filtering
    full_colors = np.zeros((len(query_points), 3))
    # Grey for Shell
    full_colors[is_shell] = [0.7, 0.7, 0.7] 
    # Orange for Infill
    full_colors[is_core & keep_infill] = [1.0, 0.7, 0.0] 
    
    valid_colors = full_colors[final_mask]
    
    print(f"  -> Generated {len(valid_points)} simulation voxels.")

    # 6. Create Visual Geometry (PointCloud representing Voxels)
    # Using a PointCloud is faster than creating thousands of cube meshes
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    
    # To look like voxels, we can optionally use o3d.geometry.VoxelGrid.create_from_point_cloud
    # but that re-discretizes. Let's return the PointCloud, it looks good dense.
    
    # Alternative: Create VoxelGrid object from points
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=res)
    
    return vg

def carve_callback(vis):
    """
    Triggered by 'C' key.
    Pauses and asks for Slicing parameters in the console.
    """
    if len(MESHES) == 0:
        print("No model to carve.")
        return False

    print("\n" + "="*40)
    print("      SLICER SIMULATION CONFIG      ")
    print("="*40)
    print("Please enter settings in the terminal:")
    
    # Pause visualizer? Open3D doesn't easily pause the run loop from a callback,
    # but input() blocks the thread, effectively pausing it.
    
    try:
        lh = get_input_default("Layer Height (mm)", 0.2)
        nw = get_input_default("Nozzle Width (mm)", 0.4)
        wc = get_input_default("Wall Count", 3, int)
        ip = get_input_default("Infill %", 20, float)
        
        print("Infill Types: [g]yroid, [h]oneycomb, [t]ri-hexagon, [l]ines")
        it_raw = input("Infill Type [gyroid]: ").strip().lower()
        
        type_map = {'g': 'gyroid', 'h': 'honeycomb', 't': 'tri-hexagon', 'l': 'triangles'}
        it = type_map.get(it_raw[0] if it_raw else 'g', 'gyroid')
        
        print(f"\nSimulating: {it} infill @ {ip}% density...")
        
        # Perform Carving on the first mesh
        sim_geo = perform_carving_tensor(MESHES[0], lh, nw, wc, ip, it)
        
        if sim_geo:
            # Remove original mesh
            vis.remove_geometry(MESHES[0], reset_bounding_box=False)
            
            # Add new simulation geometry
            vis.add_geometry(sim_geo, reset_bounding_box=False)
            
            # Update global ref so we don't carve the carve
            MESHES[0] = sim_geo
            
            print("Simulation Complete. Resume viewer.")
            return True
            
    except Exception as e:
        print(f"Carving failed: {e}")
        
    return False

def main():
    global HOLE_LOCATIONS
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="STL Viewer", width=800, height=600, left=50, top=50)

    # Register Keys
    vis.register_key_callback(65, add_file_callback) # 'A'
    vis.register_key_callback(67, carve_callback)    # 'C'

    print("Visualizer initialized.")
    print("Controls:")
    print("  - 'A': Add another STL (Screw)")
    print("  - 'C': Carve/Simulate Printing (Console Input)")
    print("  - Left Mouse: Rotate")
    print("  - Ctrl + Left Mouse: Pan")
    print("  - 'Q': Quit")

    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not initial_path:
        print("Select the first file (Base object)...")
        initial_path = get_file_path()
    
    if initial_path:
        mesh = load_mesh(initial_path)
        if mesh:
            mesh.paint_uniform_color([0.6, 0.6, 0.6]) # Grey
            MESHES.append(mesh)
            HOLE_LOCATIONS = detect_hole_locations(mesh)
            vis.add_geometry(mesh)
        else:
            print("Failed to load initial mesh. Press 'A' to try again.")
    else:
        print("No initial file loaded. Press 'A' to load a file.")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
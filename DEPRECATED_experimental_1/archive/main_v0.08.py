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
# State for Cross Section View
IS_SECTION_VIEW = False
# Store original Z-near to restore after section view
ORIGINAL_Z_NEAR = -1.0

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
    
    A = np.column_stack((x, y, np.ones(len(x))))
    B = x**2 + y**2
    
    try:
        solution, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        D, E, F = solution
        
        xc = D / 2.0
        yc = E / 2.0
        return np.array([xc, yc])
    except Exception as e:
        print(f"     -> Circle fit failed: {e}")
        return np.mean(points_2d, axis=0)

def detect_hole_locations(mesh):
    """
    Analyzes the base mesh to find vertical holes on the top surface.
    """
    print("Scanning base mesh for holes...")
    
    aabb = mesh.get_axis_aligned_bounding_box()
    max_z = aabb.get_max_bound()[2]
    
    pts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    is_shelf = (pts[:, 2] < (max_z - 1.0)) & (normals[:, 2] > 0.9)
    shelf_pts = pts[is_shelf]
    
    z_threshold = max_z - 25.0 
    
    mask = (pts[:, 2] > z_threshold) & (np.abs(normals[:, 2]) < 0.2)
    wall_pts = pts[mask]
    
    if len(wall_pts) == 0:
        print("  -> No vertical wall geometry found in top section.")
        return [aabb.get_center()]
        
    pts_2d = wall_pts[:, :2]
    
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
        
        min_b = np.min(cluster_pts, axis=0)
        max_b = np.max(cluster_pts, axis=0)
        dims = max_b - min_b
        diagonal = np.linalg.norm(dims)
        
        if len(cluster_pts) > 3:
            center_xy = fit_circle_least_squares(cluster_pts)
        else:
            center_xy = np.mean(cluster_pts, axis=0)

        target_z = max_z 
        
        if len(shelf_pts) > 0:
            dx = shelf_pts[:, 0] - center_xy[0]
            dy = shelf_pts[:, 1] - center_xy[1]
            dist_sq = dx*dx + dy*dy
            
            nearby_shelf_indices = np.where(dist_sq < 144.0)[0] 
            
            if len(nearby_shelf_indices) > 0:
                nearby_z = shelf_pts[nearby_shelf_indices, 2]
                target_z = np.median(nearby_z)
                print(f"     Feature {i}: Found internal shelf at Z={target_z:.2f}")
        
        center_3d = np.array([center_xy[0], center_xy[1], target_z])
        
        if diagonal > 8.0 and diagonal < 45.0:
            candidates.append(center_3d)
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> CANDIDATE (Precise Center)")
        else:
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> Ignored (Size)")
            
    candidates.sort(key=lambda p: (p[0], p[1]))
    
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
    
    num_slices = 100
    step = height / num_slices
    
    top_threshold = max_z - (height * 0.05)
    top_mask = (z > top_threshold)
    
    if not np.any(top_mask): 
        return max_z
        
    top_pts = pts[top_mask]
    top_width = np.max(top_pts[:,0]) - np.min(top_pts[:,0])
    top_depth = np.max(top_pts[:,1]) - np.min(top_pts[:,1])
    head_diameter = (top_width + top_depth) / 2.0
    
    for i in range(num_slices):
        z_high = max_z - (i * step)
        z_low = z_high - step
        
        mask = (z >= z_low) & (z < z_high)
        if not np.any(mask): continue
        
        slice_pts = pts[mask]
        w = np.max(slice_pts[:,0]) - np.min(slice_pts[:,0])
        d = np.max(slice_pts[:,1]) - np.min(slice_pts[:,1])
        slice_dia = (w + d) / 2.0
        
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
    
    if len(HOLE_LOCATIONS) == 0:
        target_pos = base.get_axis_aligned_bounding_box().get_center()
        print("  -> No holes detected, using base center.")
    else:
        target_idx = hole_index % len(HOLE_LOCATIONS)
        target_pos = HOLE_LOCATIONS[target_idx]
        print(f"  -> Targeting Hole {target_idx + 1} at {target_pos[:2]}")

    screw_center = screw.get_axis_aligned_bounding_box().get_center()
    dx = target_pos[0] - screw_center[0]
    dy = target_pos[1] - screw_center[1]
    
    screw.translate([dx, dy, 0])

    shoulder_z = find_screw_shoulder_z(screw)
    base_shelf_z = target_pos[2] 
    
    dz = base_shelf_z - shoulder_z
    screw.translate([0, 0, dz])
    
    print("  -> Fit complete.")

def add_file_callback(vis):
    """
    Callback function triggered by pressing 'A'.
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
        if len(MESHES) > 0:
            print("Detected additional mesh (Screw).")
            base_mesh = MESHES[0]
            mesh.paint_uniform_color([0.8, 0.2, 0.2]) # Red
            fit_screw_to_hole(mesh, base_mesh, SCREW_COUNT)
            SCREW_COUNT += 1

        MESHES.append(mesh)
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
    try:
        val = input(f"{prompt} [{default_val}]: ").strip()
        if not val:
            return default_val
        return cast_type(val)
    except Exception:
        return default_val

def generate_gyroid(x, y, z, scale):
    return (np.sin(x * scale) * np.cos(y * scale) +
            np.sin(y * scale) * np.cos(z * scale) +
            np.sin(z * scale) * np.cos(x * scale))

def generate_honeycomb(x, y, z, scale):
    return (np.sin(x*scale) + np.sin((x*0.5 + y*0.866)*scale) + 
            np.sin((x*0.5 - y*0.866)*scale))

def perform_carving_tensor(mesh, layer_h, nozzle_w, walls, infill_pct, infill_type):
    """
    Uses Open3D Tensor API (RaycastingScene) to compute the carved geometry.
    Returns a TRIANGLE MESH if scikit-image is available (Isosurface Extraction).
    Returns a VOXEL GRID if not.
    """
    print("\n--- Processing Geometry (Calculated Isosurface) ---")
    
    # Check for optional dependencies for Mathematical Mesh Generation
    try:
        from skimage import measure
        has_skimage = True
    except ImportError:
        print("Warning: 'scikit-image' not found. Falling back to blocky voxels.")
        print("To get smooth mathematical surfaces, run: pip install scikit-image")
        has_skimage = False

    try:
        import open3d.core as o3c
        from open3d.t.geometry import RaycastingScene, TriangleMesh
    except ImportError:
        print("Error: Open3D Tensor API not available.")
        return None

    # 1. Setup Raycasting Scene
    t_mesh = TriangleMesh.from_legacy(mesh)
    scene = RaycastingScene()
    scene.add_triangles(t_mesh)
    
    # 2. Define Query Grid
    # ULTRA-HIGH RESOLUTION MODE
    # As requested: min(layer_h, nozzle_w) / 4.0
    # This ignores computational cost.
    res = min(layer_h, nozzle_w) / 4.0
    
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()
    dims = max_b - min_b
    
    print(f"  -> High-Fidelity Mode Active. Voxel Size: {res:.4f}mm")
    
    # Calculate expected voxel count for warning
    expected_voxels = (dims[0]/res) * (dims[1]/res) * (dims[2]/res)
    if expected_voxels > 1e8: # > 100 Million
        print(f"  -> WARNING: This will require massive memory (~{expected_voxels/1e6:.1f}M points).")
        print("  -> Preparing waffle iron...")

    xs = np.arange(min_b[0], max_b[0], res)
    ys = np.arange(min_b[1], max_b[1], res)
    zs = np.arange(min_b[2], max_b[2], res)
    
    # Create the 3D grid
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
    query_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1).astype(np.float32)
    
    print(f"  -> Computing math for {len(query_points)} points...")
    
    query_tensor = o3c.Tensor(query_points, dtype=o3c.float32)

    # 3. Compute SDF
    signed_dist = scene.compute_signed_distance(query_tensor)
    sdf_vals = signed_dist.numpy()
    
    # --- FIX for Inverted Normals ---
    # Open3D convention: Inside < 0, Outside > 0.
    # If the mesh has inverted normals, Inside > 0, Outside < 0.
    # We check the corners of the bounding box (query grid). These are almost always outside.
    # If corners are NEGATIVE, the SDF is inverted (Outside is negative).
    # We flip the sign to normalize it.
    if sdf_vals.size > 0:
        # Check a few corners to be sure (0=min corner, -1=max corner)
        if sdf_vals[0] < 0 and sdf_vals[-1] < 0:
            print("  -> Detected inverted SDF (Outside is Negative). Flipping sign.")
            sdf_vals = -sdf_vals

    # 4. Define Regions
    wall_thickness = walls * nozzle_w
    
    is_shell = (sdf_vals <= 0) & (sdf_vals > -wall_thickness)
    is_core = sdf_vals <= -wall_thickness
    
    # 5. Infill Pattern
    scale_factor = (infill_pct / 20.0) * (2.0 * np.pi / 10.0)
    
    if infill_type == 'gyroid':
        pattern_vals = generate_gyroid(query_points[:,0], query_points[:,1], query_points[:,2], scale_factor)
        keep_infill = pattern_vals > 0
    elif infill_type == 'honeycomb':
        pattern_vals = generate_honeycomb(query_points[:,0], query_points[:,1], query_points[:,2], scale_factor)
        keep_infill = pattern_vals > 0
    elif infill_type == 'tri-hexagon':
        v1 = np.sin(query_points[:,0]*scale_factor)
        v2 = np.sin((query_points[:,0]*0.5 + query_points[:,1]*0.866)*scale_factor)
        keep_infill = (v1 * v2) > 0
    else: 
        v1 = np.sin(query_points[:,0]*scale_factor)
        v2 = np.sin(query_points[:,1]*scale_factor)
        keep_infill = (v1 + v2) > 0.5
        
    # The final boolean mask (True = Material, False = Air)
    final_mask = is_shell | (is_core & keep_infill)
    
    # --- BRANCH: Marching Cubes (Smooth) vs Voxels (Blocky) ---
    
    if has_skimage:
        print("  -> Generating smooth isosurface (Marching Cubes)...")
        
        # Reshape mask to 3D volume
        # We need a float volume for marching cubes. 
        # Ideally we use the actual SDF values combined, but boolean is 'ok' for "hard" slicing
        volume = final_mask.reshape(xv.shape).astype(float)
        
        # Run Marching Cubes
        # level=0.5 picks the boundary between True (1.0) and False (0.0)
        try:
            verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, spacing=(res, res, res))
            
            # Vertices are returned in local coordinates (0 to size*spacing). 
            # We must offset them by the grid origin (min_b)
            verts += min_b
            
            # Create Open3D Mesh
            mesh_out = o3d.geometry.TriangleMesh()
            mesh_out.vertices = o3d.utility.Vector3dVector(verts)
            mesh_out.triangles = o3d.utility.Vector3iVector(faces)
            
            # Visuals
            mesh_out.compute_vertex_normals()
            mesh_out.paint_uniform_color([1.0, 0.7, 0.0]) # Orange-ish for printed look
            
            print(f"  -> Generated Mesh with {len(verts)} vertices.")
            return mesh_out
            
        except Exception as e:
            print(f"  -> Marching cubes failed ({e}), falling back to voxels.")

    # Fallback / Voxel Mode
    valid_points = query_points[final_mask]
    full_colors = np.zeros((len(query_points), 3))
    full_colors[is_shell] = [0.7, 0.7, 0.7] 
    full_colors[is_core & keep_infill] = [1.0, 0.7, 0.0] 
    valid_colors = full_colors[final_mask]
    
    print(f"  -> Generated {len(valid_points)} voxels.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=res)
    
    return vg

def carve_callback(vis):
    """
    Triggered by 'C' key.
    """
    if len(MESHES) == 0:
        print("No model to carve.")
        return False

    print("\n" + "="*40)
    print("      SLICER SIMULATION CONFIG      ")
    print("="*40)
    print("Please enter settings in the terminal:")
    
    try:
        lh = get_input_default("Layer Height (mm)", 0.2)
        nw = get_input_default("Nozzle Width (mm)", 0.4)
        wc = get_input_default("Wall Count", 3, int)
        ip = get_input_default("Infill %", 20, float)
        
        print("Infill Types: [g]yroid, [h]oneycomb, [t]ri-hexagon, [l]ines")
        it_raw = input("Infill Type [gyroid]: ").strip().lower()
        
        type_map = {'g': 'gyroid', 'h': 'honeycomb', 't': 'tri-hexagon', 'l': 'triangles'}
        it = type_map.get(it_raw[0] if it_raw else 'g', 'gyroid')
        
        print(f"\nSimulating {it} infill...")
        
        sim_geo = perform_carving_tensor(MESHES[0], lh, nw, wc, ip, it)
        
        if sim_geo:
            vis.remove_geometry(MESHES[0], reset_bounding_box=False)
            vis.add_geometry(sim_geo, reset_bounding_box=False)
            MESHES[0] = sim_geo
            
            print("Simulation Complete. Resume viewer.")
            return True
            
    except Exception as e:
        print(f"Carving failed: {e}")
        
    return False

def toggle_section_view(vis):
    """
    Triggered by 'I' key.
    Toggles interactive cross-section view using Z-Near clipping.
    """
    global IS_SECTION_VIEW
    IS_SECTION_VIEW = not IS_SECTION_VIEW
    
    ctr = vis.get_view_control()
    
    if IS_SECTION_VIEW:
        print("\n[I] Cross-Section Mode: ON")
        print("    -> Drag MOUSE to Rotate/Orbit (Plane follows camera)")
        print("    -> ZOOM (Ctrl+Drag/Scroll) to move Plane depth")
        
        # Calculate exact distance to center
        if len(MESHES) > 0:
            center = MESHES[0].get_axis_aligned_bounding_box().get_center()
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            
            extrinsic = np.asarray(cam_params.extrinsic)
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            # Camera Position in World Coords
            cam_pos = -np.linalg.inv(R) @ t
            
            dist_to_center = np.linalg.norm(cam_pos - center)
            
            # IMPORTANT: For "cut in half", we set the near clip plane 
            # to be exactly at the center.
            ctr.set_constant_z_near(dist_to_center)
            print(f"    -> Setting Clip Plane at distance: {dist_to_center:.2f}")
        else:
            ctr.set_constant_z_near(10.0)
            
        vis.update_renderer() # Force update to see the cut
        
    else:
        print("\n[I] Cross-Section Mode: OFF")
        # Reset clipping
        ctr.set_constant_z_near(-1.0)
        vis.update_renderer()
    
    return True

def main():
    global HOLE_LOCATIONS
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="STL Viewer", width=800, height=600, left=50, top=50)

    # Register Keys
    vis.register_key_callback(65, add_file_callback) # 'A'
    vis.register_key_callback(67, carve_callback)    # 'C'
    vis.register_key_callback(73, toggle_section_view) # 'I'

    print("Visualizer initialized.")
    print("Controls:")
    print("  - 'A': Add another STL (Screw)")
    print("  - 'C': Carve/Simulate Printing")
    print("  - 'I': Toggle Cross-Section (Move/Rotate with Mouse)")
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
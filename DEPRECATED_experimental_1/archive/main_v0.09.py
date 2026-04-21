import open3d as o3d
import sys
import os
import subprocess
import numpy as np
import copy
import math
import gc # Garbage Collection for memory optimization

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
        try:
            script = """
            set theFile to choose file with prompt "Select an STL file" of type {"stl"}
            POSIX path of theFile
            """
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None 
        except Exception as e:
            print(f"Error opening macOS file picker: {e}")
            return None
    else:
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
        return np.mean(points_2d, axis=0)

def detect_hole_locations(mesh):
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
        
        center_3d = np.array([center_xy[0], center_xy[1], target_z])
        if diagonal > 8.0 and diagonal < 45.0:
            candidates.append(center_3d)
            
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
    
    if len(final_candidates) == 0:
         return [aabb.get_center()]
    return final_candidates

def find_screw_shoulder_z(mesh):
    pts = np.asarray(mesh.vertices)
    if len(pts) == 0: return 0
    z = pts[:, 2]
    min_z, max_z = np.min(z), np.max(z)
    height = max_z - min_z
    num_slices = 100
    step = height / num_slices
    top_threshold = max_z - (height * 0.05)
    top_mask = (z > top_threshold)
    if not np.any(top_mask): return max_z
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
            return z_high
    return min_z 

def fit_screw_to_hole(screw, base, hole_index):
    global HOLE_LOCATIONS
    print(f"Fitting Screw #{hole_index + 1}...")
    if len(HOLE_LOCATIONS) == 0:
        target_pos = base.get_axis_aligned_bounding_box().get_center()
    else:
        target_idx = hole_index % len(HOLE_LOCATIONS)
        target_pos = HOLE_LOCATIONS[target_idx]

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
    global SCREW_COUNT
    print("\n--- Add File Triggered ---")
    file_path = get_file_path()
    if not file_path: return False
    print(f"Loading {file_path}...")
    mesh = load_mesh(file_path)
    if mesh:
        if len(MESHES) > 0:
            print("Detected additional mesh (Screw).")
            base_mesh = MESHES[0]
            mesh.paint_uniform_color([0.8, 0.2, 0.2]) 
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
#  TOOLPATH-STYLE CARVING (Fast & Low RAM)
# ==========================================

def get_input_default(prompt, default_val, cast_type=float):
    try:
        val = input(f"{prompt} [{default_val}]: ").strip()
        if not val:
            return default_val
        return cast_type(val)
    except Exception:
        return default_val

def generate_gyroid_batch(x, y, z, scale):
    """Optimized vector math for gyroid"""
    return (np.sin(x * scale) * np.cos(y * scale) +
            np.sin(y * scale) * np.cos(z * scale) +
            np.sin(z * scale) * np.cos(x * scale))

def generate_honeycomb_batch(x, y, z, scale):
    return (np.sin(x*scale) + np.sin((x*0.5 + y*0.866)*scale) + 
            np.sin((x*0.5 - y*0.866)*scale))

def perform_carving_toolpath(mesh, layer_h, nozzle_w, walls, infill_pct, infill_type):
    """
    VECTOR / TOOLPATH SIMULATION
    Replacement for mesh.section() using robust SDF slicing.
    This calculates layer-by-layer to mimic a slicer without RAM explosion.
    """
    print("\n--- Processing Geometry (Vector-First Toolpath Mode) ---")
    
    try:
        import open3d.core as o3c
        from open3d.t.geometry import RaycastingScene, TriangleMesh
    except ImportError:
        print("Error: Open3D Tensor API not available.")
        return None

    # 1. Setup Scene (for efficient Inside/Outside checks)
    t_mesh = TriangleMesh.from_legacy(mesh)
    scene = RaycastingScene()
    scene.add_triangles(t_mesh)
    
    # 2. Setup Slicing
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()
    
    # Use standard slicing resolution logic
    zs = np.arange(min_b[2], max_b[2], layer_h)
    print(f"  -> Slicing {len(zs)} Layers...")

    # Grid Resolution: High enough to look like a solid line/path
    line_res = nozzle_w / 2.0  
    xs = np.arange(min_b[0], max_b[0], line_res)
    ys = np.arange(min_b[1], max_b[1], line_res)
    
    # Infill Settings
    scale_factor = (infill_pct / 20.0) * (2.0 * np.pi / 10.0)
    wall_thickness = walls * nozzle_w 

    infill_func = None
    if infill_type == 'gyroid': infill_func = generate_gyroid_batch
    elif infill_type == 'honeycomb': infill_func = generate_honeycomb_batch
    elif infill_type == 'tri-hexagon': 
        def tri_hex(x, y, z, s): return np.sin(x*s) * np.sin((x*0.5 + y*0.866)*s)
        infill_func = tri_hex
    else: 
        def lines(x, y, z, s): return (np.sin(x*s) + np.sin(y*s)) - 0.5
        infill_func = lines

    shell_points_store = [] 
    infill_points_store = [] 
    
    # Pre-calculate 2D grid for one layer (optimization)
    xv_layer, yv_layer = np.meshgrid(xs, ys, indexing='ij')
    xv_flat = xv_layer.flatten().astype(np.float32)
    yv_flat = yv_layer.flatten().astype(np.float32)

    total_elements = 0
    
    for i, z_val in enumerate(zs):
        if i % 10 == 0:
            print(f"     Slicing Layer {i}/{len(zs)}...")
            gc.collect() 

        # Create 3D points for this layer
        zv_flat = np.full_like(xv_flat, z_val)
        layer_points = np.stack([xv_flat, yv_flat, zv_flat], axis=1)
        
        # --- A. WALLS & SHELL (Via SDF) ---
        # Since mesh.section() failed, we compute SDF on the grid.
        # This is fast for a single layer.
        query_tensor = o3c.Tensor(layer_points, dtype=o3c.float32)
        signed_dist = scene.compute_signed_distance(query_tensor)
        sdf_vals = signed_dist.numpy()
        
        # Handle Inverted Normals (Check corners)
        if sdf_vals.size > 0 and sdf_vals[0] < 0 and sdf_vals[-1] < 0:
             sdf_vals = -sdf_vals
             
        # Identify Shell (Walls)
        # Points inside the object but close to surface
        is_shell = (sdf_vals <= 0) & (sdf_vals > -wall_thickness)
        is_core = sdf_vals <= -wall_thickness
        
        valid_shell = layer_points[is_shell]
        if len(valid_shell) > 0:
            shell_points_store.append(valid_shell)
            total_elements += len(valid_shell)

        # --- B. INFILL (Pattern First) ---
        # Filter Core points for infill pattern
        if np.any(is_core):
            core_indices = np.where(is_core)[0]
            core_points = layer_points[core_indices]
            
            # Apply Pattern Math
            pattern_vals = infill_func(core_points[:,0], core_points[:,1], core_points[:,2], scale_factor)
            keep_infill = pattern_vals > 0
            
            valid_infill = core_points[keep_infill]
            if len(valid_infill) > 0:
                infill_points_store.append(valid_infill)
                total_elements += len(valid_infill)
                
        # Cleanup
        del layer_points, query_tensor, signed_dist, sdf_vals

    print(f"  -> Assembly: {total_elements:,} elements generated.")
    
    # --- GEOMETRY CREATION ---
    final_geoms = []
    
    if shell_points_store:
        all_shell = np.concatenate(shell_points_store, axis=0)
        pcd_shell = o3d.geometry.PointCloud()
        pcd_shell.points = o3d.utility.Vector3dVector(all_shell)
        pcd_shell.paint_uniform_color([0.7, 0.7, 0.7]) # Grey Walls
        final_geoms.append(pcd_shell)
        
    if infill_points_store:
        all_infill = np.concatenate(infill_points_store, axis=0)
        pcd_infill = o3d.geometry.PointCloud()
        pcd_infill.points = o3d.utility.Vector3dVector(all_infill)
        pcd_infill.paint_uniform_color([1.0, 0.6, 0.0]) # Orange Infill
        final_geoms.append(pcd_infill)

    if not final_geoms:
        print("Error: No geometry generated.")
        return None
        
    return final_geoms

def carve_callback(vis):
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
        it = {'g': 'gyroid', 'h': 'honeycomb', 't': 'tri-hexagon', 'l': 'triangles'}.get(it_raw[0] if it_raw else 'g', 'gyroid')
        
        print(f"\nSimulating {it} infill (Vector Toolpath Mode)...")
        
        sim_geoms = perform_carving_toolpath(MESHES[0], lh, nw, wc, ip, it)
        
        if sim_geoms:
            vis.remove_geometry(MESHES[0], reset_bounding_box=False)
            
            for geom in sim_geoms:
                vis.add_geometry(geom, reset_bounding_box=False)
            
            MESHES[0] = sim_geoms[0] 
            
            opt = vis.get_render_option()
            opt.point_size = 2.0 # Finer points for infill
            
            print("Simulation Complete. Resume viewer.")
            return True
            
    except Exception as e:
        print(f"Carving failed: {e}")
        import traceback
        traceback.print_exc()
        
    return False

def toggle_section_view(vis):
    global IS_SECTION_VIEW
    IS_SECTION_VIEW = not IS_SECTION_VIEW
    ctr = vis.get_view_control()
    
    if IS_SECTION_VIEW:
        print("\n[I] Cross-Section Mode: ON")
        if len(MESHES) > 0:
            center = MESHES[0].get_axis_aligned_bounding_box().get_center()
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = np.asarray(cam_params.extrinsic)
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            cam_pos = -np.linalg.inv(R) @ t
            dist_to_center = np.linalg.norm(cam_pos - center)
            ctr.set_constant_z_near(dist_to_center)
        else:
            ctr.set_constant_z_near(10.0)
    else:
        print("\n[I] Cross-Section Mode: OFF")
        ctr.set_constant_z_near(-1.0)
    
    return True

def main():
    global HOLE_LOCATIONS
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="STL Viewer", width=800, height=600, left=50, top=50)

    vis.register_key_callback(65, add_file_callback) # 'A'
    vis.register_key_callback(67, carve_callback)    # 'C'
    vis.register_key_callback(73, toggle_section_view) # 'I'

    print("Visualizer initialized.")
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not initial_path:
        print("Select the first file (Base object)...")
        initial_path = get_file_path()
    
    if initial_path:
        mesh = load_mesh(initial_path)
        if mesh:
            mesh.paint_uniform_color([0.6, 0.6, 0.6]) 
            MESHES.append(mesh)
            HOLE_LOCATIONS = detect_hole_locations(mesh)
            vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
import open3d as o3d
import sys
import os
import subprocess
import numpy as np
import copy
import math
import gc 

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

# --- SLICER STATE MANAGEMENT ---
class SlicerState:
    def __init__(self):
        self.active = False
        self.mesh = None
        self.scene = None
        self.zs = [] 
        self.current_layer_idx = 0
        
        # Chunking State
        self.xv_flat = None
        self.yv_flat = None
        self.total_2d_points = 0
        self.chunk_ptr = 0
        self.CHUNK_SIZE = 200000 
        
        # Params
        self.wall_thickness = 0
        self.scale_factor = 0
        self.infill_func = None
        self.invert_sdf = False 
        self.resolution = 0.1 # Default resolution for normal estimation
        
        # Visuals
        self.vis = None 
        
        # Optimization:
        # Instead of one growing PCD, we keep the raw data in lists
        # and add small temporary PCDs to the viewer for speed.
        self.storage_points = [] 
        self.storage_colors = []
        self.temp_geometries = [] # Track what we added to scene to remove later

SLICER = SlicerState()

# Only import tkinter if NOT on macOS to avoid NSApplication crashes
if sys.platform != 'darwin':
    import tkinter as tk
    from tkinter import simpledialog, filedialog

def get_file_path():
    """
    Opens a file picker dialog.
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

def get_slicing_params_gui():
    """
    Opens a robust, platform-appropriate dialog for Slicing Parameters.
    """
    defaults = "0.2, 0.4, 3, 20, gyroid"
    prompt = "Enter Settings:\\nLayer Height (mm), Nozzle (mm), Walls, Infill %, Pattern"
    
    input_str = None
    
    if sys.platform == 'darwin':
        try:
            script = f"""
            display dialog "{prompt}" default answer "{defaults}" with title "Slicer Settings" buttons {{"Cancel", "Slice"}} default button "Slice"
            text returned of result
            """
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode == 0:
                input_str = result.stdout.strip()
        except Exception as e:
            print(f"Dialog error: {e}")
            return None
    else:
        try:
            root = tk.Tk()
            root.withdraw()
            root.update()
            input_str = simpledialog.askstring("Slicer Settings", 
                                             "Enter: LayerH, NozzleW, Walls, Infill%, Pattern\n(comma separated)", 
                                             initialvalue=defaults)
            root.update()
            root.destroy()
        except Exception:
            return None

    if not input_str:
        return None

    try:
        parts = [p.strip() for p in input_str.split(',')]
        lh = float(parts[0])
        nw = float(parts[1])
        wc = int(parts[2])
        ip = float(parts[3])
        it_raw = parts[4].lower() if len(parts) > 4 else 'gyroid'
        type_map = {'g': 'gyroid', 'h': 'honeycomb', 't': 'tri-hexagon', 'l': 'triangles'}
        it = type_map.get(it_raw[0], 'gyroid')
        return (lh, nw, wc, ip, it)
    except Exception as e:
        print(f"Invalid input format: {e}")
        return None

def load_mesh(file_path):
    if not file_path or not os.path.exists(file_path): return None
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_triangles(): return None
        mesh.compute_vertex_normals()
        return mesh
    except Exception: return None

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
    except Exception:
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
    
    if len(wall_pts) == 0: return [aabb.get_center()]
        
    pts_2d = wall_pts[:, :2]
    pcd_2d = o3d.geometry.PointCloud()
    pts_3d_for_clustering = np.zeros((len(pts_2d), 3))
    pts_3d_for_clustering[:, :2] = pts_2d
    pcd_2d.points = o3d.utility.Vector3dVector(pts_3d_for_clustering)
    
    labels = np.array(pcd_2d.cluster_dbscan(eps=2.5, min_points=10, print_progress=False))
    if len(labels) == 0: return [aabb.get_center()]

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
    
    if len(final_candidates) == 0: return [aabb.get_center()]
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
#  REAL-TIME SLICER ENGINE (Optimized)
# ==========================================

def generate_gyroid_batch(x, y, z, scale):
    return (np.sin(x * scale) * np.cos(y * scale) +
            np.sin(y * scale) * np.cos(z * scale) +
            np.sin(z * scale) * np.cos(x * scale))

def generate_honeycomb_batch(x, y, z, scale):
    return (np.sin(x*scale) + np.sin((x*0.5 + y*0.866)*scale) + 
            np.sin((x*0.5 - y*0.866)*scale))

def slicer_animation_step(vis):
    """
    NON-BLOCKING CHUNKED PROCESSING
    Calculates a small part of a layer, updates the view, then yields.
    Keeps the OS Event Loop alive so Alt-Tab works.
    """
    global SLICER
    
    if not SLICER.active:
        return False
        
    if SLICER.current_layer_idx >= len(SLICER.zs):
        print("Slicing Complete. Merging geometry...")
        SLICER.active = False
        vis.register_animation_callback(None) 
        
        # --- FINAL MERGE ---
        # 1. Clear temporary chunks
        for geom in SLICER.temp_geometries:
            vis.remove_geometry(geom, reset_bounding_box=False)
        SLICER.temp_geometries.clear()
        
        # 2. Build final single PCD
        if SLICER.storage_points:
            master_pts = np.concatenate(SLICER.storage_points, axis=0)
            master_cols = np.concatenate(SLICER.storage_colors, axis=0)
            
            final_pcd = o3d.geometry.PointCloud()
            final_pcd.points = o3d.utility.Vector3dVector(master_pts)
            final_pcd.colors = o3d.utility.Vector3dVector(master_cols)
            
            # --- FIX: Compute Normals for Final Object ---
            try:
                # Radius = ~3x grid resolution ensures we catch neighbors
                final_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=SLICER.resolution * 3.0, max_nn=30))
            except Exception:
                pass # Fallback to flat if too sparse
            
            vis.add_geometry(final_pcd, reset_bounding_box=False)
            
            # Update global reference
            MESHES[0] = final_pcd
            
        print("Final geometry ready.")
        return False

    # --- PROCESS ONE CHUNK ---
    
    # 1. Determine Range
    start = SLICER.chunk_ptr
    end = min(SLICER.total_2d_points, start + SLICER.CHUNK_SIZE)
    
    if start >= SLICER.total_2d_points:
        # Layer Done, move to next
        SLICER.current_layer_idx += 1
        SLICER.chunk_ptr = 0
        
        if SLICER.current_layer_idx % 5 == 0:
            gc.collect()
            print(f"Completed Layer {SLICER.current_layer_idx}/{len(SLICER.zs)}")
            
        return True 

    # 2. Get Coordinates
    z_val = SLICER.zs[SLICER.current_layer_idx]
    
    xv_chunk = SLICER.xv_flat[start:end]
    yv_chunk = SLICER.yv_flat[start:end]
    zv_chunk = np.full_like(xv_chunk, z_val)
    
    chunk_points = np.stack([xv_chunk, yv_chunk, zv_chunk], axis=1)
    
    # 3. Calculate SDF
    try:
        import open3d.core as o3c
        query_tensor = o3c.Tensor(chunk_points, dtype=o3c.float32)
        signed_dist = SLICER.scene.compute_signed_distance(query_tensor)
        sdf_vals = signed_dist.numpy()
        
        if SLICER.invert_sdf:
             sdf_vals = -sdf_vals
             
        is_shell = (sdf_vals <= 0) & (sdf_vals > -SLICER.wall_thickness)
        is_core = sdf_vals <= -SLICER.wall_thickness
        
        chunk_pts_found = []
        chunk_cols_found = []
        
        # Shell
        valid_shell = chunk_points[is_shell]
        if len(valid_shell) > 0:
            chunk_pts_found.append(valid_shell)
            c = np.zeros((len(valid_shell), 3))
            c[:] = [0.7, 0.7, 0.7]
            chunk_cols_found.append(c)

        # Infill
        if np.any(is_core):
            core_indices = np.where(is_core)[0]
            core_points = chunk_points[core_indices]
            pattern_vals = SLICER.infill_func(core_points[:,0], core_points[:,1], core_points[:,2], SLICER.scale_factor)
            keep_infill = pattern_vals > 0
            
            valid_infill = core_points[keep_infill]
            if len(valid_infill) > 0:
                chunk_pts_found.append(valid_infill)
                c = np.zeros((len(valid_infill), 3))
                c[:] = [1.0, 0.6, 0.0]
                chunk_cols_found.append(c)
        
        # 4. Update Buffers & Visuals
        if chunk_pts_found:
            new_pts = np.concatenate(chunk_pts_found, axis=0)
            new_cols = np.concatenate(chunk_cols_found, axis=0)
            
            # Store for End-of-Print merge
            SLICER.storage_points.append(new_pts)
            SLICER.storage_colors.append(new_cols)
            
            # Create TEMPORARY geometry for animation speed
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(new_pts)
            temp_pcd.colors = o3d.utility.Vector3dVector(new_cols)
            
            # --- FIX: Compute Normals for Animation Chunks ---
            try:
                temp_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=SLICER.resolution * 3.0, max_nn=30))
            except Exception:
                pass
            
            vis.add_geometry(temp_pcd, reset_bounding_box=False)
            SLICER.temp_geometries.append(temp_pcd)

    except Exception as e:
        print(f"Slice error: {e}")
        SLICER.active = False
        return False

    # 5. Advance Pointer
    SLICER.chunk_ptr = end
    
    # Force event processing
    vis.poll_events()
    vis.update_renderer()
    
    return True 

def init_slicing_mode(vis, mesh, layer_h, nozzle_w, walls, infill_pct, infill_type):
    global SLICER
    print("\n--- Initializing Slicer ---")
    
    try:
        import open3d.core as o3c
        from open3d.t.geometry import RaycastingScene, TriangleMesh
    except ImportError:
        print("Error: Open3D Tensor API not available.")
        return False

    t_mesh = TriangleMesh.from_legacy(mesh)
    SLICER.scene = RaycastingScene()
    SLICER.scene.add_triangles(t_mesh)
    
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()
    
    # Global Inversion Check
    test_p = min_b - [1.0, 1.0, 1.0] 
    test_tensor = o3c.Tensor([test_p], dtype=o3c.float32)
    dist = SLICER.scene.compute_signed_distance(test_tensor).numpy()
    
    if dist[0] < 0:
        print("  -> Detected globally inverted normals. Correcting...")
        SLICER.invert_sdf = True
    else:
        SLICER.invert_sdf = False
    
    SLICER.zs = np.arange(min_b[2], max_b[2], layer_h)
    print(f"  -> Total Layers: {len(SLICER.zs)}")

    line_res = nozzle_w / 2.0  
    
    # Store resolution for Normals
    SLICER.resolution = line_res 
    
    xs = np.arange(min_b[0], max_b[0], line_res)
    ys = np.arange(min_b[1], max_b[1], line_res)
    xv_layer, yv_layer = np.meshgrid(xs, ys, indexing='ij')
    
    SLICER.xv_flat = xv_layer.flatten().astype(np.float32)
    SLICER.yv_flat = yv_layer.flatten().astype(np.float32)
    SLICER.total_2d_points = len(SLICER.xv_flat)
    
    SLICER.wall_thickness = walls * nozzle_w 
    SLICER.scale_factor = (infill_pct / 20.0) * (2.0 * np.pi / 10.0)
    
    if infill_type == 'gyroid': SLICER.infill_func = generate_gyroid_batch
    elif infill_type == 'honeycomb': SLICER.infill_func = generate_honeycomb_batch
    elif infill_type == 'tri-hexagon': 
        def tri_hex(x, y, z, s): return np.sin(x*s) * np.sin((x*0.5 + y*0.866)*s)
        SLICER.infill_func = tri_hex
    else: 
        def lines(x, y, z, s): return (np.sin(x*s) + np.sin(y*s)) - 0.5
        SLICER.infill_func = lines

    # HIDE OLD MESH
    vis.remove_geometry(mesh, reset_bounding_box=False)
    
    # Init storage
    SLICER.storage_points = []
    SLICER.storage_colors = []
    SLICER.temp_geometries = []
    
    opt = vis.get_render_option()
    opt.point_size = 2.0 
    
    # Reset State
    SLICER.current_layer_idx = 0
    SLICER.chunk_ptr = 0
    SLICER.active = True
    vis.register_animation_callback(slicer_animation_step)
    
    return True

def carve_callback(vis):
    if len(MESHES) == 0:
        print("No model to carve.")
        return False

    params = get_slicing_params_gui()
    if not params: return False
    lh, nw, wc, ip, it = params
    print(f"\nStarting Real-Time Print Simulation...")
    print(f"Layer={lh}mm, Infill={ip}%, Pattern={it}")
    
    init_slicing_mode(vis, MESHES[0], lh, nw, wc, ip, it)
    # Ref updates happen at end of animation now
    return True

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
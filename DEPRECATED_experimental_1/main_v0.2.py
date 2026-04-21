import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import os
import subprocess
import numpy as np
import copy
import math
import gc 
import json
import collections

# ==========================================
#  GLOBAL MATH & LOGIC FUNCTIONS (Unchanged)
# ==========================================

def load_materials_db():
    try:
        with open('materials.json', 'r') as f:
            return json.load(f)
    except:
        return {}

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

def validate_hole_candidate(center_xy, points_2d, normals_2d):
    # Vector from center to point
    vecs = points_2d - center_xy
    dists = np.linalg.norm(vecs, axis=1)
    valid_pts = dists > 1e-6
    if np.sum(valid_pts) < 3: return False

    vecs = vecs[valid_pts]
    normals = normals_2d[valid_pts]
    dists = dists[valid_pts]

    # Normalize vectors
    vecs_normalized = vecs / dists[:, np.newaxis]

    # Normalize normals (they are 2D now)
    norms_mag = np.linalg.norm(normals, axis=1)
    valid_norms = norms_mag > 1e-6
    if np.sum(valid_norms) < 3: return False

    vecs_normalized = vecs_normalized[valid_norms]
    normals = normals[valid_norms] / norms_mag[valid_norms][:, np.newaxis]
    dists = dists[valid_norms]

    # 1. Check Normal Direction
    # For a hole (concave), normal points towards center.
    # Vector P-C (vecs_normalized) points away from center.
    # Dot product should be -1.
    dots = np.sum(vecs_normalized * normals, axis=1)
    avg_dot = np.mean(dots)

    # We expect close to -1. If it's > -0.5, it's likely not a hole (could be flat wall or stud).
    if avg_dot > -0.5:
        return False

    # 2. Check Circularity (Residuals)
    mean_r = np.mean(dists)
    std_r = np.std(dists)

    if mean_r > 0:
        if (std_r / mean_r) > 0.15: # 15% deviation allowed
            return False

    # 3. Check Angular Coverage
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    angles = np.sort(angles)
    gaps = np.diff(angles)
    if len(angles) > 1:
        last_gap = (angles[0] + 2*np.pi) - angles[-1]
        gaps = np.append(gaps, last_gap)
        max_gap = np.max(gaps)

        # Require at least 200 degrees coverage (max gap < 160 degrees)
        if max_gap > np.deg2rad(160):
            return False
    else:
        return False

    return True

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
    wall_normals = normals[mask]
    
    if len(wall_pts) == 0: return [aabb.get_center()]
        
    pts_2d = wall_pts[:, :2]
    normals_2d = wall_normals[:, :2]
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
        if len(cluster_pts) > 3:
            center_xy = fit_circle_least_squares(cluster_pts)
        else:
            center_xy = np.mean(cluster_pts, axis=0)

        # Validation
        cluster_normals = normals_2d[cluster_indices]
        if not validate_hole_candidate(center_xy, cluster_pts, cluster_normals):
            continue

        target_z = max_z 
        if len(shelf_pts) > 0:
            dx = shelf_pts[:, 0] - center_xy[0]
            dy = shelf_pts[:, 1] - center_xy[1]
            dist_sq = dx*dx + dy*dy
            nearby_shelf_indices = np.where(dist_sq < 144.0)[0] 
            if len(nearby_shelf_indices) > 0:
                nearby_z = shelf_pts[nearby_shelf_indices, 2]
                target_z = np.median(nearby_z)
        
        # Dimensions check
        min_b = np.min(cluster_pts, axis=0)
        max_b = np.max(cluster_pts, axis=0)
        dims = max_b - min_b
        diagonal = np.linalg.norm(dims)
        
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

def generate_gyroid_batch(x, y, z, scale):
    return (np.sin(x * scale) * np.cos(y * scale) +
            np.sin(y * scale) * np.cos(z * scale) +
            np.sin(z * scale) * np.cos(x * scale))

def generate_honeycomb_batch(x, y, z, scale):
    return (np.sin(x*scale) + np.sin((x*0.5 + y*0.866)*scale) + 
            np.sin((x*0.5 - y*0.866)*scale))

def create_pbr_material(color, metallic=0.0, roughness=0.5, reflectance=0.5):
    """
    Creates a PBR material with the given properties.
    """
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = list(color)
    mat.base_metallic = metallic
    mat.base_roughness = roughness
    mat.base_reflectance = reflectance
    return mat

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

# ==========================================
#  SLICING LOGIC
# ==========================================

class Slicer:
    def __init__(self, app, mesh, nozzle_width, layer_height, wall_count, infill_percent):
        self.app = app
        self.mesh = mesh
        try:
            self.nozzle_width = float(nozzle_width)
            self.layer_height = float(layer_height)
            self.wall_count = int(wall_count)
            self.infill = float(infill_percent)
        except ValueError:
            print("Invalid slice settings.")
            self.app.info_label.text = "Invalid slice settings."
            return

        # Setup Raycasting Scene
        try:
            # Create Tensor Mesh
            self.t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(self.t_mesh)
        except Exception as e:
            print(f"Failed to initialize Slicer Raycasting: {e}")
            self.app.info_label.text = f"Slicer Init Error: {e}"
            return

        # Calculate bounds and steps
        bbox = mesh.get_axis_aligned_bounding_box()
        self.min_bound = bbox.get_min_bound()
        self.max_bound = bbox.get_max_bound()

        self.min_z = self.min_bound[2]
        self.max_z = self.max_bound[2]
        self.height = self.max_z - self.min_z
        self.num_layers = int(np.ceil(self.height / self.layer_height))

        # Grid setup
        # Resolution determined by nozzle width
        padding = self.nozzle_width * 2
        self.grid_min = self.min_bound - padding
        self.grid_max = self.max_bound + padding

        width_x = self.grid_max[0] - self.grid_min[0]
        width_y = self.grid_max[1] - self.grid_min[1]

        self.res_x = int(np.ceil(width_x / self.nozzle_width))
        self.res_y = int(np.ceil(width_y / self.nozzle_width))

        # Pre-compute query grid coordinates (X, Y)
        xs = np.linspace(self.grid_min[0], self.grid_max[0], self.res_x)
        ys = np.linspace(self.grid_min[1], self.grid_max[1], self.res_y)
        self.xx, self.yy = np.meshgrid(xs, ys)

        self.current_layer = 0
        self.accumulated_points = []

        # Cache for Phase 2: Stores (mask, eroded_mask, z) for each layer
        self.cached_layers = []

        # Phases: 'WALLS' (Outline only) -> 'INFILL' (Full with Infill)
        self.phase = "WALLS"

        # Hide original base mesh
        if self.app.widget3d.scene.has_geometry("Base"):
             self.app.widget3d.scene.show_geometry("Base", False)

        print(f"Starting slice: {self.num_layers} layers.")
        self.app.lbl_progress.text = "Generating Walls..."
        self.app.progress_panel.visible = True
        self.app.window.set_needs_layout()

        gui.Application.instance.post_to_main_thread(self.app.window, self.step)

    def step(self):
        # Handle Phase Transitions
        if self.current_layer >= self.num_layers:
            if self.phase == "WALLS":
                # Finish Phase 1 -> Start Phase 2
                print("Walls done. Starting Infill Phase.")
                self.phase = "INFILL"
                self.current_layer = 0

                # Clear visual geometries from Phase 1
                for i in range(self.num_layers):
                    if self.app.widget3d.scene.has_geometry(f"Slice_{i}"):
                         self.app.widget3d.scene.remove_geometry(f"Slice_{i}")

                # Reset accumulation for the final build
                self.accumulated_points = []

                self.app.lbl_progress.text = f"Generating Infill 0/{self.num_layers}"
                # Yield to let UI update (clear screen)
                gui.Application.instance.post_to_main_thread(self.app.window, self.step)
                return
            else:
                # Finish Phase 2 -> Done
                self.finish()
                return

        # Update Progress Text during Infill
        if self.phase == "INFILL":
            if self.current_layer % 5 == 0: # Update every few layers to save UI calls
                self.app.lbl_progress.text = f"Generating Infill {self.current_layer}/{self.num_layers}"

        # Common variables
        z = 0.0
        final_mask = None
        eroded_mask = None # Needed for infill calculation
        mask = None
        is_solid_layer = (self.current_layer < self.wall_count) or \
                         (self.current_layer >= (self.num_layers - self.wall_count))

        if self.phase == "WALLS":
            # --- PHASE 1: Heavy Computation (Occupancy + Erosion) ---

            # Calculate Z for this layer
            z = self.min_z + (self.current_layer * self.layer_height) + (self.layer_height / 2.0)

            # Create Query Points (H, W, 3)
            zz = np.full_like(self.xx, z)
            query_points = np.stack([self.xx, self.yy, zz], axis=-1).astype(np.float32)

            # Compute Occupancy
            flat_query = query_points.reshape((-1, 3))
            tensor_query = o3d.core.Tensor(flat_query)

            occupancy = self.scene.compute_occupancy(tensor_query)
            mask_flat = occupancy.numpy().astype(bool)
            mask = mask_flat.reshape((self.res_y, self.res_x))

            if is_solid_layer:
                final_mask = mask
                # Solid layer implies full fill, so 'eroded' effectively empty or we just use mask for infill later if needed
                eroded_mask = np.zeros_like(mask)
            else:
                # Hollow Layer: Wall = Mask - Eroded(Mask)
                if np.any(mask):
                    eroded = mask.copy()
                    for _ in range(self.wall_count):
                        # Manual erosion
                        padded = np.pad(eroded, 1, mode='constant', constant_values=0)
                        eroded = eroded & padded[0:-2, 1:-1] & padded[2:, 1:-1] & \
                                 padded[1:-1, 0:-2] & padded[1:-1, 2:]

                    # Wall is Mask minus Interior(Eroded)
                    final_mask = mask & (~eroded)
                    eroded_mask = eroded # Store for infill usage
                else:
                    final_mask = mask
                    eroded_mask = np.zeros_like(mask)

            # Cache the results for Phase 2
            self.cached_layers.append({
                'mask': mask,
                'eroded_mask': eroded_mask,
                'final_mask': final_mask, # Phase 1 only shows walls
                'z': z
            })

        else:
            # --- PHASE 2: Lightweight Infill Generation (Using Cache) ---

            # Retrieve cached data
            if self.current_layer < len(self.cached_layers):
                data = self.cached_layers[self.current_layer]
                mask = data['mask']
                eroded_mask = data['eroded_mask']
                # Start with the walls from Phase 1
                final_mask = data['final_mask'].copy()
                z = data['z']
            else:
                # Should not happen
                print(f"Error: Missing cache for layer {self.current_layer}")
                self.current_layer += 1
                gui.Application.instance.post_to_main_thread(self.app.window, self.step)
                return

            # Apply Infill Logic
            if (not is_solid_layer) and (self.infill > 0) and np.any(eroded_mask):
                # Calculate Infill Mask
                # Scale: Map 10% -> 1.0, 20% -> 2.0 approx?
                # Heuristic: scale = 0.5 + (infill / 5.0).
                scale = 0.5 + (self.infill / 5.0)

                # Optimization: Only compute gyroid where needed
                valid_indices = np.where(eroded_mask)
                if len(valid_indices[0]) > 0:
                    gx = self.xx[valid_indices]
                    gy = self.yy[valid_indices]
                    gz = np.full_like(gx, z)

                    gyroid_vals = generate_gyroid_batch(gx, gy, gz, scale)

                    # Threshold > 0 gives ~50% density relative to the cell size
                    infill_hits = gyroid_vals > 0

                    # Create a temporary full mask for infill
                    infill_full_mask = np.zeros_like(mask)
                    infill_full_mask[valid_indices] = infill_hits

                    final_mask = final_mask | infill_full_mask

        # Extract Points for Visualization (and Accumulation in Phase 2)
        if np.any(final_mask):

            # We need coordinates. In Phase 2 we have z, but need to reconstruct query_points
            # or just use indices on meshgrid.
            # Reconstructing query_points is fast.
            zz = np.full_like(self.xx, z)
            # Use stack only on valid indices to save memory? No, simpler to index into meshgrid

            valid_indices = np.where(final_mask)

            # Extract x,y,z
            px = self.xx[valid_indices]
            py = self.yy[valid_indices]
            pz = zz[valid_indices]

            layer_points = np.stack([px, py, pz], axis=-1).astype(np.float32)

            # Store (only in INFILL phase matters for final mesh)
            if self.phase == "INFILL":
                self.accumulated_points.append(layer_points)

            # Update Visualization (Show slice)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(layer_points)

            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            # Color: Orange for walls (Phase 1), slightly different or same for Phase 2?
            # Let's use Orange for consistency.
            mat.base_color = [1.0, 0.5, 0.0, 1.0]

            layer_name = f"Slice_{self.current_layer}"
            self.app.widget3d.scene.add_geometry(layer_name, pcd, mat)

        self.current_layer += 1

        # Yield to UI
        gui.Application.instance.post_to_main_thread(self.app.window, self.step)

    def finish(self):
        self.app.lbl_progress.text = "Finalizing Mesh... (May take a moment)"
        # Force a redraw so the user sees the message before the heavy mesh reconstruction starts
        self.app.window.set_needs_layout()
        self.app.window.post_redraw()

        # Schedule the actual work in the next cycle to allow the UI to update
        gui.Application.instance.post_to_main_thread(self.app.window, self._finish_worker)

    def _finish_worker(self):
        print("Slicing finished.")
        if not self.accumulated_points:
             self.app.info_label.text = "Slicing failed (no points)."
             self.app.progress_panel.visible = False
             return

        all_pts = np.concatenate(self.accumulated_points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.estimate_normals()

        try:
            print("Reconstructing mesh...")
            # Alpha shape is approximate but better than nothing
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=self.nozzle_width * 2.0)
            mesh.compute_vertex_normals()

            if len(mesh.vertices) == 0:
                raise Exception("Empty mesh generated")

            # Clean up visualization
            for i in range(self.num_layers):
                if self.app.widget3d.scene.has_geometry(f"Slice_{i}"):
                    self.app.widget3d.scene.remove_geometry(f"Slice_{i}")

            # Update Base Mesh
            self.app.meshes[0] = mesh

            mat = create_pbr_material([0.1, 0.6, 0.6, 1.0], metallic=0.0, roughness=0.3)

            # Update Scene
            if self.app.widget3d.scene.has_geometry("Base"):
                 self.app.widget3d.scene.remove_geometry("Base")

            try:
                t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                self.app.widget3d.scene.add_geometry("Base", t_mesh, mat)
            except:
                self.app.widget3d.scene.add_geometry("Base", mesh, mat)

            self.app.info_label.text = "Slicing Complete."
            print("Slicing Complete.")
        except Exception as e:
            print(f"Mesh reconstruction failed: {e}. Keeping Point Cloud.")
            self.app.info_label.text = "Slicing Complete (View Only)."

        self.app.progress_panel.visible = False
        self.app.btn_load_screw.enabled = True
        self.app.window.post_redraw()

# ==========================================
#  MODERN GUI APPLICATION (CLEAN)
# ==========================================

class SimToolApp:
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = gui.Application.instance.create_window("SimTool Pro (Viewer)", 1024, 768)
        self.w = self.window 
        
        # --- INITIALIZE LOGIC STATE ---
        self.meshes = [] # [0]=Base, [1+]=Screws
        self.hole_locations = []
        self.screw_count = 0
        self.materials_db = load_materials_db()
        
        # Slice setup state
        self.slice_inputs = ["Nozzle Width", "Layer Height", "Wall Count", "Infill %"]
        self.slice_values = []
        self.slice_step = 0
        self.is_slice_setup_active = False

        # --- 3D SCENE WIDGET ---
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.9, 0.9, 0.9, 1.0])
        
        # Use Hard Shadows for better depth definition
        self.widget3d.scene.set_lighting(rendering.Open3DScene.LightingProfile.HARD_SHADOWS, (0.577, -0.577, -0.577))
        
        # Enable Ambient Occlusion (SSAO) and Antialiasing
        try:
            if hasattr(self.widget3d.scene, "view"):
                self.widget3d.scene.view.set_ambient_occlusion(True, ssct_enabled=True)
                self.widget3d.scene.view.set_antialiasing(True)
                print("Enabled SSAO and Antialiasing")
        except Exception as e:
            print(f"Could not enable advanced rendering features: {e}")

        # Optional: Try-catch for low level shadow enables if they exist
        try:
            if hasattr(self.widget3d.scene.scene, "enable_sun_shadows"):
                self.widget3d.scene.scene.enable_sun_shadows(True)
        except Exception:
            pass

        # FIX: Disable caching to ensure rendering updates correctly on interaction
        self.widget3d.enable_scene_caching(False)
        
        # Add widget directly to window (No Layout/Sidebar)
        self.window.add_child(self.widget3d)

        # --- UI OVERLAY (Top Right) ---
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        # Semi-transparent white background
        self.panel.background_color = gui.Color(1, 1, 1, 0.8)

        self.lbl_holes = gui.Label("Holes Detected: 0")
        self.panel.add_child(self.lbl_holes)

        self.btn_load_screw = gui.Button("Load Screw")
        self.btn_load_screw.set_on_clicked(self.on_load_screw)
        self.panel.add_child(self.btn_load_screw)

        # Slice Setup Button
        self.btn_slice_setup = gui.Button("Slice Setup")
        self.btn_slice_setup.set_on_clicked(self.start_slice_setup)
        self.btn_slice_setup.enabled = False
        self.panel.add_child(self.btn_slice_setup)

        self.window.add_child(self.panel)
        
        # --- INFO PANEL (Bottom Left) ---
        self.info_panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.info_panel.background_color = gui.Color(1, 1, 1, 0.8)
        self.info_label = gui.Label("")
        self.info_panel.add_child(self.info_label)
        self.info_panel.visible = False
        self.window.add_child(self.info_panel)

        # --- SLICE SETUP OVERLAY (Centered) ---
        self.slice_overlay = gui.Vert(0.5 * em, gui.Margins(1 * em, 1 * em, 1 * em, 1 * em))
        self.slice_overlay.background_color = gui.Color(0.95, 0.95, 0.95, 1.0)

        self.lbl_slice_prompt = gui.Label("Enter Value:")
        self.slice_overlay.add_child(self.lbl_slice_prompt)

        self.txt_slice_input = gui.TextEdit()
        # We need to capture Enter. TextEdit might not bubble it up.
        # We will check keys in _on_key or add a button if needed.
        self.slice_overlay.add_child(self.txt_slice_input)

        # Adding a Next button just in case Enter is hard to capture robustly
        self.btn_slice_next = gui.Button("Next")
        self.btn_slice_next.set_on_clicked(self.on_slice_next)
        self.slice_overlay.add_child(self.btn_slice_next)

        self.slice_overlay.visible = False
        self.window.add_child(self.slice_overlay)

        # --- PROGRESS PANEL (Top Center) ---
        self.progress_panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.progress_panel.background_color = gui.Color(1, 1, 1, 0.8)
        self.lbl_progress = gui.Label("")
        self.progress_panel.add_child(self.lbl_progress)
        self.progress_panel.visible = False
        self.window.add_child(self.progress_panel)

        # Hook layout to resize widget to window
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        
        # Load initial file if arg provided
        slicing_params = None
        args = sys.argv[1:]

        if len(args) >= 5:
            # Check for slicing params at the end
            try:
                # float, float, int, float
                p1 = float(args[-4])
                p2 = float(args[-3])
                p3 = int(args[-2])
                p4 = float(args[-1])
                slicing_params = [args[-4], args[-3], args[-2], args[-1]]

                base_path = args[0]
                screw_paths = args[1:-4]
            except ValueError:
                # Not params, assume all are screws
                base_path = args[0]
                screw_paths = args[1:]
        elif len(args) >= 1:
            base_path = args[0]
            screw_paths = args[1:]
        else:
            base_path = None
            screw_paths = []

        if base_path:
            self.load_mesh(base_path)
            for sp in screw_paths:
                self.load_screw(sp)

            if slicing_params:
                print(f"Auto-starting slice with params: {slicing_params}")
                self.run_slicing_operation(*slicing_params)
        else:
            # Open dialog immediately if no args
            path = get_file_path()
            if path:
                self.load_mesh(path)

    def _on_layout(self, layout_context):
        # Resize 3D widget to fill the entire window
        r = self.window.content_rect
        self.widget3d.frame = r
        em = self.window.theme.font_size

        # Resize Top-Right Panel
        panel_width = 15 * em
        ctx = gui.Widget.Constraints()
        ctx.width = panel_width
        panel_height = self.panel.calc_preferred_size(layout_context, ctx).height
        self.panel.frame = gui.Rect(r.width - panel_width - em, em, panel_width, panel_height)

        # Resize Bottom-Left Info Panel
        if self.info_panel.visible:
            info_size = self.info_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            self.info_panel.frame = gui.Rect(em, r.height - info_size.height - em, info_size.width, info_size.height)

        # Resize Progress Panel (Top Center)
        if self.progress_panel.visible:
            prog_size = self.progress_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
            cx = r.width / 2.0
            self.progress_panel.frame = gui.Rect(cx - prog_size.width / 2.0, em, prog_size.width, prog_size.height)

        # Resize Slice Overlay (Centered)
        if self.slice_overlay.visible:
            overlay_size = self.slice_overlay.calc_preferred_size(layout_context, gui.Widget.Constraints())
            # Ensure minimum width
            overlay_width = max(overlay_size.width, 15 * em)
            overlay_height = overlay_size.height

            cx = r.width / 2.0
            cy = r.height / 2.0

            self.slice_overlay.frame = gui.Rect(cx - overlay_width/2.0, cy - overlay_height/2.0, overlay_width, overlay_height)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.UP:
            if event.key == gui.KeyName.A:
                # Only load screw if not disabled
                if self.btn_load_screw.enabled:
                    self.on_load_screw()
                return True

            if event.key == gui.KeyName.S:
                if self.btn_slice_setup.enabled and not self.is_slice_setup_active:
                    self.start_slice_setup()
                return True

            if self.is_slice_setup_active:
                if event.key == gui.KeyName.ENTER:
                    self.on_slice_next()
                    return True

        return False

    def update_slice_ui_state(self):
        # Check if we have enough screws
        # Note: len(self.meshes) includes base mesh at index 0
        screw_count = len(self.meshes) - 1
        holes_count = len(self.hole_locations)

        can_slice = (screw_count >= holes_count) and (holes_count > 0)
        self.btn_slice_setup.enabled = can_slice

        # Update button text/color to indicate readiness?
        # Default styling is fine for now.

    def start_slice_setup(self):
        self.is_slice_setup_active = True
        self.slice_values = []
        self.slice_step = 0
        self.slice_overlay.visible = True

        self.update_slice_prompt()
        self.window.set_needs_layout()

    def update_slice_prompt(self):
        if self.slice_step < len(self.slice_inputs):
            prompt = self.slice_inputs[self.slice_step]
            self.lbl_slice_prompt.text = f"Enter {prompt}:"
            self.txt_slice_input.text_value = ""
            self.window.set_focus_widget(self.txt_slice_input)
        else:
            self.finish_slice_setup()

    def on_slice_next(self):
        if not self.is_slice_setup_active:
            return

        val = self.txt_slice_input.text_value.strip()
        if not val:
            # Maybe show error or ignore
            return

        self.slice_values.append(val)
        self.slice_step += 1

        if self.slice_step < len(self.slice_inputs):
            self.update_slice_prompt()
        else:
            self.finish_slice_setup()

    def run_slicing_operation(self, nozzle, layer, walls, infill):
        # Display results in bottom left
        info_text = "Slicing in progress...\n"
        vals = [nozzle, layer, walls, infill]
        for i, val in enumerate(vals):
            info_text += f"{self.slice_inputs[i]}: {val}\n"

        self.info_label.text = info_text
        self.info_panel.visible = True

        # Disable Load Screw
        self.btn_load_screw.enabled = False
        self.window.set_needs_layout()

        # Start Slicer
        try:
            if len(self.meshes) > 0:
                self.slicer = Slicer(
                    self,
                    self.meshes[0],
                    nozzle,
                    layer,
                    walls,
                    infill
                )
        except Exception as e:
            print(f"Error starting slicer: {e}")
            self.info_label.text = f"Error: {e}"

    def finish_slice_setup(self):
        self.is_slice_setup_active = False
        self.slice_overlay.visible = False

        if len(self.slice_values) == 4:
            self.run_slicing_operation(
                self.slice_values[0],
                self.slice_values[1],
                self.slice_values[2],
                self.slice_values[3]
            )

    def on_load_screw(self):
        path = get_file_path()
        if path:
            self.load_screw(path)

    def load_screw(self, path):
        print(f"Loading screw from: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_triangles():
            print("Failed to load screw.")
            return
        mesh.compute_vertex_normals()

        self.meshes.append(mesh)

        # Metal-like material for screw
        mat = create_pbr_material([0.7, 0.7, 0.7, 1.0], metallic=0.8, roughness=0.2, reflectance=0.9)

        name = f"Screw_{len(self.meshes)-1}"

        try:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.widget3d.scene.add_geometry(name, t_mesh, mat)
        except Exception:
             self.widget3d.scene.add_geometry(name, mesh, mat)

        # Update UI State
        self.update_slice_ui_state()

        self.window.post_redraw()

    def load_mesh(self, path):
        print(f"Loading mesh from: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_triangles(): 
            print("Failed to load mesh or empty mesh.")
            return
        mesh.compute_vertex_normals()
        
        # Debug: Print Bounds
        bbox = mesh.get_axis_aligned_bounding_box()
        print(f"Mesh Bounds: {bbox.get_min_bound()} to {bbox.get_max_bound()}")
        
        # Setup Material: Plastic-like
        # Using a slightly lower roughness and some reflectance gives it volume.
        mat = create_pbr_material([0.1, 0.6, 0.6, 1.0], metallic=0.0, roughness=0.3, reflectance=0.5)
        
        self.meshes = [mesh] # Reset
        
        # Reset Scene
        self.widget3d.scene.clear_geometry()
        
        # Re-apply environment settings after clear (crucial for some backends)
        self.widget3d.scene.set_background([0.9, 0.9, 0.9, 1.0])

        # Use Hard Shadows
        self.widget3d.scene.set_lighting(rendering.Open3DScene.LightingProfile.HARD_SHADOWS, (0.577, -0.577, -0.577))

        # Re-enable SSAO if scene was cleared (view properties might persist, but safe to re-check)
        try:
            if hasattr(self.widget3d.scene, "view"):
                self.widget3d.scene.view.set_ambient_occlusion(True, ssct_enabled=True)
                self.widget3d.scene.view.set_antialiasing(True)
        except Exception:
            pass
        
        # Convert to Tensor Geometry
        try:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self.widget3d.scene.add_geometry("Base", t_mesh, mat)
        except Exception as e:
            print(f"Tensor conversion failed, using legacy: {e}")
            self.widget3d.scene.add_geometry("Base", mesh, mat)
        
        # Setup Camera
        self.widget3d.setup_camera(60, bbox, bbox.get_center())
        
        # Detect Holes (Calculation only)
        self.hole_locations = detect_hole_locations(mesh)
        self.lbl_holes.text = f"Holes Detected: {len(self.hole_locations)}"

        # Reset Slice UI
        self.btn_load_screw.enabled = True
        self.info_panel.visible = False
        self.update_slice_ui_state()
        
        self.window.post_redraw()

    def run(self):
        self.app.run()

if __name__ == "__main__":
    app = SimToolApp()
    app.run()
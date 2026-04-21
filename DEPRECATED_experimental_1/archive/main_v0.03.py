import open3d as o3d
import sys
import os
import subprocess
import numpy as np
import copy

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

def detect_hole_locations(mesh):
    """
    Analyzes the base mesh to find vertical holes on the top surface.
    IMPROVED METHOD: Filters for Vertical Walls only (Normal Z ~ 0).
    This removes the flat top surface connecting the holes, allowing DBSCAN 
    to see them as separate rings.
    """
    print("Scanning base mesh for holes...")
    
    # 1. Get Geometry Data
    aabb = mesh.get_axis_aligned_bounding_box()
    max_z = aabb.get_max_bound()[2]
    
    pts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # 2. Filter: Top Volume + Vertical Walls Only
    # - Top 25mm of the object
    # - Normal Z component < 0.2 (Horizontal-ish normals = Vertical walls)
    # This ignores the flat top face and the flat bottom of the counterbore.
    z_threshold = max_z - 25.0 
    
    mask = (pts[:, 2] > z_threshold) & (np.abs(normals[:, 2]) < 0.2)
    wall_pts = pts[mask]
    
    if len(wall_pts) == 0:
        print("  -> No vertical wall geometry found in top section.")
        return [aabb.get_center()]
        
    # Project to XY plane for clustering
    pts_2d = wall_pts[:, :2]
    
    # 3. Cluster using DBSCAN
    # eps=2.5: Connect points within 2.5mm
    # min_points=10: Lowered slightly to catch sparser meshes
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
        
        # Calculate Centroid
        center_xy = np.mean(cluster_pts, axis=0)
        center_3d = np.array([center_xy[0], center_xy[1], max_z])
        
        # FILTERING LOGIC:
        # M10 head is ~18mm. Outer walls are usually much larger.
        # Noise is usually much smaller.
        if diagonal > 8.0 and diagonal < 45.0:
            candidates.append(center_3d)
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> CANDIDATE")
        else:
            print(f"     Feature {i}: Dia {diagonal:.1f}mm -> Ignored (Size)")
            
    # Sort candidates by X then Y (Left to Right, Front to Back)
    candidates.sort(key=lambda p: (p[0], p[1]))
    
    # Merge close duplicates (just in case)
    final_candidates = []
    if len(candidates) > 0:
        curr = candidates[0]
        group = [curr]
        
        for next_cand in candidates[1:]:
            dist = np.linalg.norm(next_cand - curr)
            if dist < 6.0: # Merging threshold
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
    base_top_z = target_pos[2] # We stored max_z in the third component
    
    dz = base_top_z - shoulder_z
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

def main():
    global HOLE_LOCATIONS
    
    # 1. Initialize Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="STL Viewer", width=800, height=600, left=50, top=50)

    # 2. Register 'A' key
    vis.register_key_callback(65, add_file_callback)

    print("Visualizer initialized.")
    print("Controls:")
    print("  - 'A' key: Add another STL file (Screw)")
    print("  - Left Mouse: Rotate")
    print("  - Ctrl + Left Mouse: Pan")
    print("  - Mouse Wheel: Zoom")
    print("  - 'Q': Quit")

    # 3. Load initial file
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not initial_path:
        print("Select the first file (Base object)...")
        initial_path = get_file_path()
    
    if initial_path:
        mesh = load_mesh(initial_path)
        if mesh:
            # Base mesh setup
            mesh.paint_uniform_color([0.6, 0.6, 0.6]) # Grey
            MESHES.append(mesh)
            
            # Detect Holes on the base mesh immediately
            HOLE_LOCATIONS = detect_hole_locations(mesh)
            
            vis.add_geometry(mesh)
        else:
            print("Failed to load initial mesh. Press 'A' to try again.")
    else:
        print("No initial file loaded. Press 'A' to load a file.")

    # 4. Run loop
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
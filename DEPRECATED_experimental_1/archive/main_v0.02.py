import open3d as o3d
import sys
import os
import subprocess
import numpy as np
import copy

# Global list to track loaded meshes for alignment
MESHES = []

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
    
    print(f"  -> Detected Head Diameter: {head_diameter:.2f}")

    # 2. Scan downwards to find where diameter drops significantly
    for i in range(num_slices):
        # Define slice window
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
            print(f"  -> Detected Shoulder at Z={z_high:.2f} (Dia: {slice_dia:.2f})")
            return z_high

    return min_z # Fallback: return bottom if no step found

def fit_screw_to_hole(screw, base):
    """
    Simple, deterministic fit:
    1. Center Screw XY to Base XY.
    2. Align Screw Shoulder (Head Bottom) to Base Top Surface.
    Assumes models are already oriented properly (Upright).
    """
    print("Auto-fitting screw (Simple Mode)...")
    
    # 1. XY Centering
    screw_center = screw.get_axis_aligned_bounding_box().get_center()
    base_center = base.get_axis_aligned_bounding_box().get_center()
    
    dx = base_center[0] - screw_center[0]
    dy = base_center[1] - screw_center[1]
    
    screw.translate([dx, dy, 0])
    print("  -> Centered XY.")

    # 2. Z Alignment (Shoulder to Top)
    # Find the "Shoulder" of the screw (bottom of the head)
    shoulder_z = find_screw_shoulder_z(screw)
    
    # Find the Top of the Base model
    base_top_z = base.get_axis_aligned_bounding_box().get_max_bound()[2]
    
    # Calculate difference and translate
    dz = base_top_z - shoulder_z
    screw.translate([0, 0, dz])
    
    print(f"  -> Aligned Screw Shoulder (Z={shoulder_z:.2f}) to Base Top (Z={base_top_z:.2f})")
    print("  -> Fit complete.")

def add_file_callback(vis):
    """
    Callback function triggered by pressing 'A'.
    Pauses the visualizer to open a file dialog and adds the new mesh.
    """
    print("\n--- Add File Triggered ---")
    
    file_path = get_file_path()
    
    if not file_path:
        print("No file selected.")
        return False
        
    print(f"Loading {file_path}...")
    mesh = load_mesh(file_path)
    
    if mesh:
        # If this is the second mesh (the screw), color it and try to fit it
        if len(MESHES) > 0:
            print("Detected second mesh (Screw). Attempting auto-fit...")
            base_mesh = MESHES[0]
            
            # Visual distinction: Make base grey, screw Red-ish
            base_mesh.paint_uniform_color([0.6, 0.6, 0.6])
            mesh.paint_uniform_color([0.8, 0.2, 0.2])
            
            # Perform the fit
            fit_screw_to_hole(mesh, base_mesh)

        # Store in global list
        MESHES.append(mesh)

        # Add to scene
        vis.add_geometry(mesh, reset_bounding_box=True)
        
        # Explicitly update
        vis.poll_events()
        vis.update_renderer()
        
        print(f"Successfully added: {os.path.basename(file_path)}")
        return True
    
    return False

def main():
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
            MESHES.append(mesh) # Add to global list
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
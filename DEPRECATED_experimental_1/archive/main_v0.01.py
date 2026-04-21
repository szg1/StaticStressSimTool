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

def align_vector_to_z(mesh):
    """
    Rotates the mesh so its longest dimension (Principal Axis) aligns with the Z-axis.
    Useful for screws which are typically long cylinders.
    """
    # 1. Compute Oriented Bounding Box to find principal axes
    obb = mesh.get_oriented_bounding_box()
    
    # 2. Rotate the mesh so its OBB aligns with the world axes
    # obb.R is the rotation matrix from local to world. R.T is world to local.
    # Applying R.T aligns the object's principal axes to X, Y, Z.
    mesh.rotate(obb.R.T, center=mesh.get_center())
    
    # 3. Ensure the longest dimension is along Z
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent() # [width, height, depth]
    
    # Find which axis is longest
    max_axis = np.argmax(extent)
    
    if max_axis == 0: # X is longest, rotate 90 deg around Y to make it Z
        R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
        mesh.rotate(R, center=mesh.get_center())
    elif max_axis == 1: # Y is longest, rotate 90 deg around X to make it Z
        R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        mesh.rotate(R, center=mesh.get_center())
    
    print("  -> Aligned longest axis to Z-axis")

def fit_screw_to_hole(screw, base):
    """
    Attempts to auto-fit the screw into the base mesh.
    1. Aligns screw to Z axis.
    2. Moves screw to the center of the base.
    3. Uses ICP to snap the screw into the hole.
    """
    print("Auto-fitting screw...")
    
    # 1. Align Screw to Vertical (Z-axis)
    align_vector_to_z(screw)
    
    # 2. Move Screw to the geometric center of the Base
    base_center = base.get_axis_aligned_bounding_box().get_center()
    screw_center = screw.get_axis_aligned_bounding_box().get_center()
    translation = base_center - screw_center
    screw.translate(translation)
    print("  -> Moved screw to base center")

    # 3. Run ICP (Iterative Closest Point) for fine-tuning
    # This "snaps" the screw into the hole if the walls are close
    threshold = 5.0 # Search radius for matching points (e.g., 5mm)
    
    # We use Point-to-Plane ICP which is robust for sliding shapes like screws/holes
    try:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            screw, base, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Apply the transformation found by ICP
        screw.transform(reg_p2l.transformation)
        print(f"  -> ICP Refinement Fitness: {reg_p2l.fitness:.4f}")
        print("  -> Screw auto-fit complete.")
    except Exception as e:
        print(f"  -> ICP failed (normals might be missing): {e}")

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
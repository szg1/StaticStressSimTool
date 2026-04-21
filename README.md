# FEA Solver for Rock Climbing Holds

This repository contains a C++ FEA (Finite Element Analysis) solver designed specifically for rock climbing holds and jugs. It simulates static stress on imported STL files.

## experimental_2 (C++)
`experimental_2` is the main project. It allows you to:
- Load a 3D model of a rock climbing hold (STL format).
- Load a screw used to fix the hold to the wall.
- Paint the pressure area where force is applied.
- Enter the applied force and simulate the resulting stress.

**Compilation Instructions**
```sh
cd experimental_2
make
```

**Usage**
```sh
./stl_viewer path/to/your/file.stl
```

**Controls**
- Left Click + Drag: Rotate
- Right Click + Drag (or Up/Down): Zoom
- Press 'A' or click "Load screw" button to add a second model.

---

## DEPRECATED_experimental_1
A deprecated Python experimental project for STL viewing and slicing.

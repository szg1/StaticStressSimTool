# Proper Readme

This repository contains a C++ STL viewer and a Python STL viewer/slicer under `experimental_2` and `DEPRECATED_experimental_1` directories respectively.

## experimental_2 (C++)
A simple STL viewer written in C++ using OpenGL and GLUT.

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

## DEPRECATED_experimental_1 (Python)
A deprecated Python experimental project for STL viewing and slicing. It uses Open3D and NumPy.

**Prerequisites**
```sh
pip install -r requirements.txt
```

**Usage**
```sh
python main_v0.2.py
```

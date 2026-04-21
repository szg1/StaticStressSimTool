CXX = g++
# Added -std=c++17 for better async/thread support and modern vector features
CXXFLAGS = -Wno-deprecated -std=c++17 
TARGET = stl_viewer

# ADDED fea_solver.cpp to the source list
SRC = main.cpp mesh.cpp button.cpp utils.cpp fea_solver.cpp

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # Accelerate is already here, which is perfect for the FEA solver
    LDFLAGS = -framework OpenGL -framework GLUT -framework Accelerate
else
    # Linux/Windows might need LAPACK/BLAS installed if you port this later
    LDFLAGS = -lGL -lGLU -lglut
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS) -pthread

clean:
	rm -f $(TARGET)
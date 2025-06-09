# SDSL - TODO list
This is a broad list of todo-tasks.
Mostly nice-to-haves, but are important for readability/usability of the package.

### Code Readability / Style
- [ ] Run auto-pep8 for all python files on build
- [ ] Add docstrings to Python methods
- [ ] Add docstrings to C++ methods
- [ ] Create Python stubs for the bindings
- [ ] Create a wiki/docs site

### Build / Install
- [ ] Make sure the Python and C++ prerequisites are minimal and clearly stated

### Utility / Clean-Code
- [ ] Add source & direction methods for R3xS2 bindings (that return numpy array)

### More Features
- [ ] Add 3D mesh environments (represented as triangle soups) -> most of the code of PCD is suitable
- [ ] Support ROS occupancy grids (maybe in a seperate sdsl-ros2 package)
- [ ] Clustering of voxels in C++
- [ ] (For comparison) implement also particle filtering method

### Completed Tasks
- [x] Create this TODO.md file
- [x] Make the install of Python bindings a single `pip install ...` command
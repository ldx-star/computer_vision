# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/237/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /snap/clion/237/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liangdaxin/computer_vision/Canny_edge_detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Canny_edge_detection.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Canny_edge_detection.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Canny_edge_detection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Canny_edge_detection.dir/flags.make

CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o: CMakeFiles/Canny_edge_detection.dir/flags.make
CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o: /home/liangdaxin/computer_vision/Canny_edge_detection/Canny.cpp
CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o: CMakeFiles/Canny_edge_detection.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o -MF CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o.d -o CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o -c /home/liangdaxin/computer_vision/Canny_edge_detection/Canny.cpp

CMakeFiles/Canny_edge_detection.dir/Canny.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Canny_edge_detection.dir/Canny.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liangdaxin/computer_vision/Canny_edge_detection/Canny.cpp > CMakeFiles/Canny_edge_detection.dir/Canny.cpp.i

CMakeFiles/Canny_edge_detection.dir/Canny.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Canny_edge_detection.dir/Canny.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liangdaxin/computer_vision/Canny_edge_detection/Canny.cpp -o CMakeFiles/Canny_edge_detection.dir/Canny.cpp.s

CMakeFiles/Canny_edge_detection.dir/main.cpp.o: CMakeFiles/Canny_edge_detection.dir/flags.make
CMakeFiles/Canny_edge_detection.dir/main.cpp.o: /home/liangdaxin/computer_vision/Canny_edge_detection/main.cpp
CMakeFiles/Canny_edge_detection.dir/main.cpp.o: CMakeFiles/Canny_edge_detection.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Canny_edge_detection.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Canny_edge_detection.dir/main.cpp.o -MF CMakeFiles/Canny_edge_detection.dir/main.cpp.o.d -o CMakeFiles/Canny_edge_detection.dir/main.cpp.o -c /home/liangdaxin/computer_vision/Canny_edge_detection/main.cpp

CMakeFiles/Canny_edge_detection.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Canny_edge_detection.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liangdaxin/computer_vision/Canny_edge_detection/main.cpp > CMakeFiles/Canny_edge_detection.dir/main.cpp.i

CMakeFiles/Canny_edge_detection.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Canny_edge_detection.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liangdaxin/computer_vision/Canny_edge_detection/main.cpp -o CMakeFiles/Canny_edge_detection.dir/main.cpp.s

CMakeFiles/Canny_edge_detection.dir/util.cpp.o: CMakeFiles/Canny_edge_detection.dir/flags.make
CMakeFiles/Canny_edge_detection.dir/util.cpp.o: /home/liangdaxin/computer_vision/Canny_edge_detection/util.cpp
CMakeFiles/Canny_edge_detection.dir/util.cpp.o: CMakeFiles/Canny_edge_detection.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Canny_edge_detection.dir/util.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Canny_edge_detection.dir/util.cpp.o -MF CMakeFiles/Canny_edge_detection.dir/util.cpp.o.d -o CMakeFiles/Canny_edge_detection.dir/util.cpp.o -c /home/liangdaxin/computer_vision/Canny_edge_detection/util.cpp

CMakeFiles/Canny_edge_detection.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Canny_edge_detection.dir/util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liangdaxin/computer_vision/Canny_edge_detection/util.cpp > CMakeFiles/Canny_edge_detection.dir/util.cpp.i

CMakeFiles/Canny_edge_detection.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Canny_edge_detection.dir/util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liangdaxin/computer_vision/Canny_edge_detection/util.cpp -o CMakeFiles/Canny_edge_detection.dir/util.cpp.s

# Object files for target Canny_edge_detection
Canny_edge_detection_OBJECTS = \
"CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o" \
"CMakeFiles/Canny_edge_detection.dir/main.cpp.o" \
"CMakeFiles/Canny_edge_detection.dir/util.cpp.o"

# External object files for target Canny_edge_detection
Canny_edge_detection_EXTERNAL_OBJECTS =

Canny_edge_detection: CMakeFiles/Canny_edge_detection.dir/Canny.cpp.o
Canny_edge_detection: CMakeFiles/Canny_edge_detection.dir/main.cpp.o
Canny_edge_detection: CMakeFiles/Canny_edge_detection.dir/util.cpp.o
Canny_edge_detection: CMakeFiles/Canny_edge_detection.dir/build.make
Canny_edge_detection: /usr/local/lib/libopencv_gapi.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_highgui.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_ml.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_objdetect.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_photo.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_stitching.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_video.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_videoio.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_dnn.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_calib3d.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_features2d.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_flann.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_imgproc.so.4.6.0
Canny_edge_detection: /usr/local/lib/libopencv_core.so.4.6.0
Canny_edge_detection: CMakeFiles/Canny_edge_detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable Canny_edge_detection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Canny_edge_detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Canny_edge_detection.dir/build: Canny_edge_detection
.PHONY : CMakeFiles/Canny_edge_detection.dir/build

CMakeFiles/Canny_edge_detection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Canny_edge_detection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Canny_edge_detection.dir/clean

CMakeFiles/Canny_edge_detection.dir/depend:
	cd /home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liangdaxin/computer_vision/Canny_edge_detection /home/liangdaxin/computer_vision/Canny_edge_detection /home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug /home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug /home/liangdaxin/computer_vision/Canny_edge_detection/cmake-build-debug/CMakeFiles/Canny_edge_detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Canny_edge_detection.dir/depend


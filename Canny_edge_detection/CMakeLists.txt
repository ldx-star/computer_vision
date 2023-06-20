cmake_minimum_required(VERSION 3.25)
project(Canny_edge_detection)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Canny_edge_detection
        Canny.cpp
        Canny.h main.cpp util.h util.cpp)
target_link_libraries(Canny_edge_detection ${OpenCV_LIBS})

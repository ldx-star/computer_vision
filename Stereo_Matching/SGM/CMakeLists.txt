cmake_minimum_required(VERSION 3.25)
project(SGM)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SGM main.cpp SGM.h SGM.cpp sgm_types.h sgm_util.h sgm_util.cpp)

target_link_libraries(SGM ${OpenCV_LIBS})
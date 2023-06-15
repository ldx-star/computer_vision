cmake_minimum_required(VERSION 3.25)
project(computer_vision)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(computer_vision
        main.cpp Vision.cpp)
target_link_libraries(computer_vision ${OpenCV_LIBS})

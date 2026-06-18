cmake_minimum_required(VERSION 3.10)
project(edge_icp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(contour_icp src/contour_icp.cpp)
target_link_libraries(contour_icp ${OpenCV_LIBS})

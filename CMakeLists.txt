cmake_minimum_required(VERSION 2.6)
project(findcontour)

add_compile_options(-std=c++11)

SET(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV 3.0 QUIET)
 if(NOT OpenCV_FOUND)
  find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# include_directories(${CMAKE_INCLUDE_DIRECTORIES}
# ${EIGEN3_INCLUDE_DIR}
# )
find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ICP ICP.cpp)
target_link_libraries(ICP ${OpenCV_LIBS})

add_executable(ppf_load_match ppf_load_match.cpp)
target_link_libraries(ppf_load_match ${OpenCV_LIBS})

add_executable (interactive_icp interactive_icp.cpp)
target_link_libraries (interactive_icp ${PCL_LIBRARIES})

add_executable (pcl_ICP pcl_ICP.cpp)
target_link_libraries (pcl_ICP ${PCL_LIBRARIES})

add_executable (usingPcl usingPcl.cpp)
target_link_libraries (usingPcl ${PCL_LIBRARIES})
target_link_libraries (usingPcl ${OpenCV_LIBS})
cmake_minimum_required(VERSION 2.6)
project(findcontour)
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

add_executable(ICP ICP.cpp)
target_link_libraries(ICP ${OpenCV_LIBS})


add_executable(ppf_load_match ppf_load_match.cpp)
target_link_libraries(ppf_load_match ${OpenCV_LIBS})

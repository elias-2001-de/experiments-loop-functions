#
# Set minimum required version
#
cmake_minimum_required(VERSION 2.8)
project(LoopFunction LANGUAGES CXX CUDA)

# Set the C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the path to the LibTorch directory
set(CMAKE_PREFIX_PATH "/home/ilyes/libtorch")

#
# Make sure we are under Unix
#
if(NOT UNIX)
  message(FATAL_ERROR "ARGoS compiles only under UNIX, sorry!")
endif(NOT UNIX)

#
# Set build options
#
include(${CMAKE_SOURCE_DIR}/src/cmake/ARGoSBuildOptions.cmake)

#
# Set compiler flags
#
include(${CMAKE_SOURCE_DIR}/src/cmake/ARGoSBuildFlags.cmake)

#
# Check for libraries
#
include(${CMAKE_SOURCE_DIR}/src/cmake/ARGoSBuildChecks.cmake)

# Include boost and zmq
include_directories(/home/igharbi/boost_1_82_0)

#
# Set base source files
#
add_subdirectory(src)

if(ARGOS_BUILD_FOR_SIMULATOR)
	add_subdirectory(loop-functions)
endif(ARGOS_BUILD_FOR_SIMULATOR)

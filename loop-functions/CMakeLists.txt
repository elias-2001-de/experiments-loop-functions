# Set up ARGoS compilation information
# Get the paths of libtorch
set(TORCH_INCLUDE_DIRS "/home/ken/depots/SwarmAC/libtorch/include" 
			"/home/ken/depots/SwarmAC/libtorch/include/torch/csrc/api/include")
set(TORCH_LIBRARY_DIRS "/home/ken/depots/SwarmAC/libtorch/lib")

# Add the include directories and library directories
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})

include_directories(/home/ken/depots/SwarmAC/argos3-nn/deps/eigen-3.4.0)
include_directories(/home/ken/depots/SwarmAC/argos3-nn/deps/MiniDNN/include)

include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIRS})
link_directories(
	${CMAKE_BINARY_DIR}/src
    )

add_subdirectory(SwarmAC)
#add_subdirectory(visual-add-ons)
# add_subdirectory(Repertoire)
# add_subdirectory(vanilla)
#add_subdirectory(chocolate)
#add_subdirectory(gianduja)
# add_subdirectory(complexity)
#add_subdirectory(extra)
# add_subdirectory(NEATvsEvo)
# add_subdirectory(AutoMoDe-Modules)
#add_subdirectory(ModularityExperiment)
#add_subdirectory(localsearch)

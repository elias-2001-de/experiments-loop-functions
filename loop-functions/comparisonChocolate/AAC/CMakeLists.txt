# Get the paths of libtorch
find_package(Torch REQUIRED)

add_library(comparison_aac_loopfunc SHARED
                          ../MADDPGLoopFunc.h
                          AACLoopFunc.h
                          AACLoopFunc.cpp
                          ${LOOP_HEADERS}
                          ${LOOP_SOURCES})

target_link_libraries(comparison_aac_loopfunc 
			${TORCH_LIBRARIES} 
			argos3plugin_${ARGOS_BUILD_FOR}_epuck)
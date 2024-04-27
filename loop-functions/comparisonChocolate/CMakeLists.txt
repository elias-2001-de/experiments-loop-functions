# Get the paths of libtorch
find_package(Torch REQUIRED)

add_library(comparison_loopfunc SHARED
                          MADDPGLoopFunc.h
                          AACLoopFunc.h
                          AACLoopFunc.cpp
                          AACLoopFuncNewObj.h
                          AACLoopFuncNewObj.cpp
                          ${LOOP_HEADERS}
                          ${LOOP_SOURCES})

target_link_libraries(comparison_loopfunc 
			${TORCH_LIBRARIES} 
			argos3plugin_${ARGOS_BUILD_FOR}_epuck)
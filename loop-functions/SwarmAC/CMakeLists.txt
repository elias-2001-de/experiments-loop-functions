find_package(cppzmq)

add_library(swarmac_loopfunc SHARED
                          SCAaroundLoopFunc.h
                          SCAaroundLoopFunc.cpp
                          Homing.h
                          Homing.cpp
                          AggregationTwoSpotsXOR2.h
                          AggregationTwoSpotsXOR2.cpp
                          ${LOOP_HEADERS}
                          ${LOOP_SOURCES})

set(TORCH_LIBRARIES "/home/ken/depots/SwarmAC/libtorch/lib/libtorch.so"
		"/home/ken/depots/SwarmAC/libtorch/lib/libtorch_cpu.so" 
		"/home/ken/depots/SwarmAC/libtorch/lib/libc10.so")

target_link_libraries(swarmac_loopfunc 
			${TORCH_LIBRARIES} 
			argos3plugin_${ARGOS_BUILD_FOR}_epuck 
			cppzmq)

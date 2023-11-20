# Get the paths of libtorch
find_package(Torch REQUIRED)
# find cppzmq
find_package(cppzmq)

add_library(swarmac_loopfunc SHARED
                        #   Homing.h
                        #   Homing.cpp
                        #   HomingTest.h
                        #   HomingTest.cpp
                        #   AggregationTwoSpotsXOR2.h
                        #   AggregationTwoSpotsXOR2.cpp
                          AACLoopFunc.h
                          AACLoopFunc.cpp
                        #   AACTestLoopFunc.h
                        #   AACTestLoopFunc.cpp
                          ${LOOP_HEADERS}
                          ${LOOP_SOURCES})

target_link_libraries(swarmac_loopfunc 
			${TORCH_LIBRARIES} 
			argos3plugin_${ARGOS_BUILD_FOR}_epuck 
			cppzmq)

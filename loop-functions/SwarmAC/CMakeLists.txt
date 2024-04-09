# Get the paths of libtorch
message(${Torch_DIR} )

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

# Set debug and release flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Conditionally add the sanitizer link library for Debug builds
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_link_libraries(swarmac_loopfunc -fsanitize=address)
endif()
find_package(Torch REQUIRED)

add_library(swarmac_loopfunc SHARED
                          AACLoopFunc.h
                          AACLoopFunc.cpp
                          SCALoopFunc.h
                          SCALoopFunc.cpp
                          Corridor.h
                          Corridor.cpp
                          ${LOOP_HEADERS}
                          ${LOOP_SOURCES})

# Set C++17 standard for the additional target
set_property(TARGET swarmac_loopfunc PROPERTY CXX_STANDARD 17)
set_property(TARGET swarmac_loopfunc PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET swarmac_loopfunc PROPERTY CXX_EXTENSIONS OFF)

target_link_libraries(swarmac_loopfunc 
			${TORCH_LIBRARIES} 
			argos3plugin_${ARGOS_BUILD_FOR}_epuck)

# Set debug and release flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Conditionally add the sanitizer link library for Debug builds
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_link_libraries(swarmac_loopfunc -fsanitize=address)
endif()
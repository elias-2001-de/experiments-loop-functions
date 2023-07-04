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

target_link_libraries(swarmac_loopfunc argos3plugin_${ARGOS_BUILD_FOR}_epuck cppzmq)

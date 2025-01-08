add_library(aggregation_loopfunc SHARED AggregationLoopFunc.h AggregationLoopFunc.cpp ${LOOP_HEADERS} ${LOOP_SOURCES})
target_link_libraries(aggregation_loopfunc argos3plugin_${ARGOS_BUILD_FOR}_epuck)

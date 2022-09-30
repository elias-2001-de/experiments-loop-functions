add_library(grid_exploration_loopfunc SHARED GridExplorationLoopFunc.h GridExplorationLoopFunc.cpp ${LOOP_HEADERS} ${LOOP_SOURCES})
target_link_libraries(grid_exploration_loopfunc argos3plugin_${ARGOS_BUILD_FOR}_epuck)

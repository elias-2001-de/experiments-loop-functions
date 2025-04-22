#pragma once

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include "../../src/CoreLoopFunctions.h"

class MarkerAggregationLoopFunction: public CoreLoopFunctions
{
public:
    MarkerAggregationLoopFunction();
    virtual ~MarkerAggregationLoopFunction();

    virtual void Destroy();
    virtual void Init(TConfigurationNode& t_tree);

    virtual argos::CColor GetFloorColor(
        argos::CVector2 const& c_position_on_plane);
    virtual void PostStep();
    virtual void Reset();

    virtual Real GetObjectiveFunction();
    virtual argos::CVector3 GetRandomPosition();

private:
    static const float _marker_radius;
    static const float _valid_area_radius;
    static const argos::CVector2 _marker_center;

    Real _objective_function;
};

inline Real MarkerAggregationLoopFunction::GetObjectiveFunction()
{
    return _objective_function;
}

#pragma once

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include "../../src/CoreLoopFunctions.h"

class NAggregationLoopFunction: public CoreLoopFunctions
{
public:
    NAggregationLoopFunction();
    virtual ~NAggregationLoopFunction();

    virtual void Destroy();
    virtual void Init(TConfigurationNode& t_tree);

    virtual argos::CColor GetFloorColor(
        argos::CVector2 const& c_position_on_plane);
    virtual void PostStep();
    virtual void Reset();

    virtual Real GetObjectiveFunction();
    virtual argos::CVector3 GetRandomPosition();

private:
    int getAreaIndex(argos::CVector2 const& position) const;

    static const int _area_count = 5;
    static const float _area_to_origin_distance;
    static const float _area_radius;

    CVector2 _area_centers[_area_count];

    Real _objective_function;
};

inline Real NAggregationLoopFunction::GetObjectiveFunction()
{
    return _objective_function;
}

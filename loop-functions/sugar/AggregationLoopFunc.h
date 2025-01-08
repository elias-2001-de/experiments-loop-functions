/**
  * @file <loop-functions/sugar/AggregationLoopFunc.h>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

#ifndef AGGREGATION_LOOP_FUNC
#define AGGREGATION_LOOP_FUNC

#include "../../src/CoreLoopFunctions.h"
#include <argos3/core/simulator/space/space.h>
#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>
#include <argos3/core/control_interface/ci_controller.h>
#include "../../../ARGoS3-AutoMoDe/src/core/AutoMoDePatchController.h"

using namespace argos;

class AggregationLoopFunction: public CoreLoopFunctions {
  public:
    AggregationLoopFunction();
    AggregationLoopFunction(const AggregationLoopFunction& orig);
    virtual ~AggregationLoopFunction();
    
    virtual void Init(argos::TConfigurationNode& t_tree);

    virtual void Destroy();
    virtual void Reset();

    virtual argos::CColor GetFloorColor(const argos::CVector2& c_position_on_plane);
    virtual void PostStep();
    

    Real GetObjectiveFunction();

    /*
     * Returns a vector containing a random position inside a circle of radius
     * m_fDistributionRadius and centered in (0,0).
     */
    virtual CVector3 GetRandomPosition();

  private:
    Real m_fRadius;
    CVector2 m_cCoordBlackSpot;
    CVector2 m_cCoordWhiteSpot;

    UInt32 m_unScoreSpot;
    Real m_fObjectiveFunction;

    std::string to_find = "patch";

    std::vector<CVector3> list_of_patch_positions = { CVector3(0, 0.6, 0), CVector3(0, -0.6, 0) };
    std::vector<UInt8> list_of_colors = { 1, 2 };
};

#endif

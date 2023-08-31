/**
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

#ifndef HomingTest
#define HomingTest

#include <torch/torch.h>

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <zmq.hpp>

#include <cmath>

#include "../../src/CoreLoopFunctions.h"
#include "../../../AC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

using namespace argos;

class HomingTest: public CoreLoopFunctions {
  public:
    HomingTest();
    HomingTest(const HomingTest& orig);
    virtual ~HomingTest();

    virtual void Destroy();

    virtual argos::CColor GetFloorColor(const argos::CVector2& c_position_on_plane);
    virtual void PostExperiment();
    virtual void PostStep();
    virtual void Reset();

    Real GetObjectiveFunction();

    CVector3 GetRandomPosition();

    void print_grid(at::Tensor grid);

  private:
    Real m_fRadius;
    CVector2 m_cCoordSpot1;
    UInt32 m_unScoreSpot1;
    Real m_fObjectiveFunction;
    Real m_fHeightNest;
};

#endif

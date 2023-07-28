/**
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

#ifndef HOMING
#define HOMING

#include <torch/torch.h>

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <zmq.hpp>

#include <cmath>

#include "../../src/CoreLoopFunctions.h"
#include "../../../AC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

using namespace argos;

class Homing: public CoreLoopFunctions {
  public:
    Homing();
    Homing(const Homing& orig);
    virtual ~Homing();

    virtual void Destroy();
    virtual void Init(TConfigurationNode& t_tree);

    virtual argos::CColor GetFloorColor(const argos::CVector2& c_position_on_plane);
    virtual void PostExperiment();
    virtual void PreStep();
    virtual void PostStep();
    virtual void Reset();

    Real GetObjectiveFunction();

    CVector3 GetRandomPosition();

    double computeBetaLogPDF(double alpha, double beta, double x);

    void update_actor(torch::nn::Module& actor_net, torch::Tensor& behavior_probs, 
                      torch::Tensor& beta_parameters, int chosen_behavior, 
                      double chosen_parameter, double td_error, 
                      torch::optim::Optimizer& optimizer);

    void print_grid(at::Tensor grid);

  private:
    Real m_fRadius;
    CVector2 m_cCoordSpot1;
    UInt32 m_unScoreSpot1;
    Real m_fObjectiveFunction;
    Real m_fHeightNest;
    
    zmq::context_t m_context;
    zmq::socket_t m_socket_actor;
    zmq::socket_t m_socket_critic;

    // Network
    CRange<Real> m_cNeuralNetworkOutputRange;
    struct Net : torch::nn::Module {
      Net() :
          fc(register_module("fc", torch::nn::Linear(2500, 1))){}

      // Implement the Net's algorithm.
      torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc(x));
	return x;
      };

      torch::nn::Linear fc;
    };
    Net critic_net; // Each thread will have its own `Net` instance
    
    // Learning variables
    float delta;
    std::vector<float> value_trace;
    std::vector<float> policy_trace;

    // State vectors
    // Vector state;         // S at step t (50x50 grid representation)
    // Vector state_prime;   // S' at step t+1
    at::Tensor state;
    at::Tensor state_prime;

    int size_policy_net = 288;
};

#endif

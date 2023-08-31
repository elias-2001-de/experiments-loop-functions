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
    int input_size = 2500;
    int hidden_size = 1;
    int output_size = 1;
    CRange<Real> m_cNeuralNetworkOutputRange;
    // Define the ConvNet architecture
    struct ConvNet : torch::nn::Module {
      ConvNet() {
        // Convolutional layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 4, 3).stride(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 8, 3).stride(1)));

        // Max pooling layers
        maxpool1 = register_module("maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3)));
        maxpool2 = register_module("maxpool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3)));

        // Fully connected layers
        fc1 = register_module("fc1", torch::nn::Linear(8 * 4 * 4, 8)); 
        fc2 = register_module("fc2", torch::nn::Linear(8, 1));
      }

      torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(maxpool1(conv1(x)));
        x = torch::relu(maxpool2(conv2(x)));
        x = x.view({x.size(0), -1}); // Flatten the tensor
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
      }


      torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
      torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr};
      torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };
    ConvNet critic_net; // Each thread will have its own `Net` instance

    // Learning variables
    float delta;
    std::vector<float> value_trace;
    std::vector<float> policy_trace;

    // State vectors
    // Vector state;         // S at step t (50x50 grid representation)
    // Vector state_prime;   // S' at step t+1
    at::Tensor state;
    at::Tensor state_prime;

    int size_value_net = 1377;
    int size_policy_net = 178;


    int gamma = 0.9;
    float lambda_critic = 0.9;
};

#endif

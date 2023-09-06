/*
 * Aggregation with Ambient Clues
 *
 * @author Antoine Ligot - <aligot@ulb.ac.be>
 */

#ifndef CHOCOLATE_AAC_LOOP_FUNC_H
#define CHOCOLATE_AAC_LOOP_FUNC_H

#include <torch/torch.h>

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <zmq.hpp>

#include <cmath>

#include "../../src/CoreLoopFunctions.h"
#include "../../../AC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

#include <typeinfo>

using namespace argos;

class AACLoopFunction : public CoreLoopFunctions {

   public:
      AACLoopFunction();
      AACLoopFunction(const AACLoopFunction& orig);
      virtual ~AACLoopFunction();

      virtual void Destroy();
      virtual void Init(TConfigurationNode& t_tree);
      virtual void Reset();
      virtual void PreStep();
      virtual void PostStep();
      virtual void PostExperiment();

      Real GetObjectiveFunction();

      virtual CColor GetFloorColor(const CVector2& c_position_on_plane);

      virtual CVector3 GetRandomPosition();

      double computeBetaLogPDF(double alpha, double beta, double x);

      void update_actor(torch::nn::Module& actor_net, torch::Tensor& behavior_probs, 
                        torch::Tensor& beta_parameters, int chosen_behavior, 
                        double chosen_parameter, double td_error, 
                        torch::optim::Optimizer& optimizer);

      void print_grid(at::Tensor grid);

    private:
      Real m_fRadius;
      CVector2 m_cCoordBlackSpot;
      CVector2 m_cCoordWhiteSpot;

      Real m_fObjectiveFunction;

      // Get the time step
      int fTimeStep;
      
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
		      conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 8).stride(4)));
		      conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 4).stride(2)));

		      // Max pooling layers
		      maxpool1 = register_module("maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3)));
		      maxpool2 = register_module("maxpool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3)));

		      // Fully connected layers
		      fc1 = register_module("fc1", torch::nn::Linear(32 * 4 * 4 + 1, 256));
		      fc2 = register_module("fc2", torch::nn::Linear(256, 1));
	      }

	      torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
          std::cout << "x size = " << x.sizes() << std::endl;
		      x = torch::relu(conv1(x));
          std::cout << "conv1(x)) size = " << x.sizes() << std::endl;
		      x = torch::relu(conv2(x));
          std::cout << "conv2(x)) size = " << x.sizes() << std::endl;
		      x = x.view({x.size(0), -1}); // Flatten the tensor
          std::cout << "x.view size = " << x.sizes() << std::endl;
		      // Concatenate x and t tensors
		      //std::cout << "x = " << x << std::endl;
		      std::cout << "t = " << t << std::endl;
		      x = torch::cat({x, t}, 1);
          std::cout << "x.cat size = " << x.sizes() << std::endl;
		      x = torch::relu(fc1->forward(x));
          std::cout << "relu((fc1(x))) size = " << x.sizes() << std::endl;
          std::cout << "fc2(x) = " << fc2->forward(x.squeeze(0)) << std::endl;
          x = fc2->forward(x);
          std::cout << "fc2(x) size = " << x.sizes() << std::endl;
		      return x;
	      }

	      torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	      torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr};
	      torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
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
      torch::Tensor time;
      torch::Tensor time_prime;

      int size_value_net = 141105;
      int size_policy_net = 96;


      float gamma = 0.99;
      float lambda_critic = 0.8;

};

#endif

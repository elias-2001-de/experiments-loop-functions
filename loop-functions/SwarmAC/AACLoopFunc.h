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

      CSpace::TMapPerType tEpuckMap;

      // Get the time step
      int fTimeStep;
      
      zmq::context_t m_context;
      zmq::socket_t m_socket_actor;
      zmq::socket_t m_socket_critic;

      // Network
      int input_size = 2;
      int hidden_size = 128;
      int output_size = 1;
      CRange<Real> m_cNeuralNetworkOutputRange;
      // Define the ConvNet architecture
      struct Net : torch::nn::Module {  
        Net(int64_t input_dim = 3, int64_t hidden_dim = 64, int64_t output_dim = 1){
          // Define the neural network layers in a Sequential module
          fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
          fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
          fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, hidden_dim));
          fc4 = register_module("fc4", torch::nn::Linear(hidden_dim, output_dim));
        }

        torch::Tensor forward(torch::Tensor x) {
          x = torch::relu(fc1->forward(x));
          x = torch::relu(fc2->forward(x));
          x = torch::relu(fc3->forward(x));
          x = fc4->forward(x);
          return x;
        }

        void print_last_layer_params() {
          std::cout << "Weights of last layer critic:\n" << fc4->weight << std::endl;
          std::cout << "Bias of last layer critic:\n" << fc4->bias << std::endl;
        }

        //torch::nn::Sequential layers;
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
      };
      Net critic_net; // Each thread will have its own `Net` instance

      // Learning variables
      float delta;
      std::vector<float> value_trace;
      std::vector<float> policy_trace;
      std::vector<float> value_update;
      std::vector<float> policy_update;

      // State vectors
      // Vector state;         // S at step t (50x50 grid representation)
      // Vector state_prime;   // S' at step t+1
      torch::Tensor state = torch::empty({3});
      torch::Tensor state_prime = torch::empty({3});
      // torch::Tensor time;
      // torch::Tensor time_prime;

      int size_value_net = 8641;
      int size_policy_net = 8836;


      float gamma = 0.99;
      float lambda_critic = 0.8;


      // adam param
      float beta1 = 0.9;   // Exponential decay rate for first moment estimate
      float beta2 = 0.999; // Exponential decay rate for second moment estimate
      float epsilon = 1e-8;
      float weight_decay = 1;
      float lr = 1e-8; // starting learning rate
      float decay_factor = 1e-5; // factor by which the learning rate will be reduced
      std::vector<float> m;
      std::vector<float> v;
      std::vector<float> v_max;

      bool decay;

};

#endif

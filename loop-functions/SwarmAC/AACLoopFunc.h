/*
 * Aggregation with Ambient Clues
 *
 * @author Antoine Ligot - <aligot@ulb.ac.be>
 */

#ifndef CHOCOLATE_AAC_LOOP_FUNC_H
#define CHOCOLATE_AAC_LOOP_FUNC_H

#include <torch/torch.h>

#include "../../src/CoreLoopFunctions.h"
#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <zmq.hpp>

#include <cmath>

#include "../../src/CoreLoopFunctions.h"
#include "../../../AC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

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

      zmq::context_t m_context;
      zmq::socket_t m_socket_actor;
      zmq::socket_t m_socket_critic;

      // Network
      int input_size = 2500;
      int hidden_size = 6;
      int output_size = 1;
      CRange<Real> m_cNeuralNetworkOutputRange;
      struct Net : torch::nn::Module {
        Net()
        : fc_input(torch::nn::LinearOptions(2500, 6).bias(true)),
          dropout_input(0.1),
          fc_output(6, 1),
          dropout_output(0.1)
        {
            register_module("fc_input", fc_input);
            register_module("dropout_input", dropout_input);	  
            register_module("fc_output", fc_output);			
            register_module("dropout_output", dropout_output);  
        }


        // Implement the Net's algorithm.
        torch::Tensor forward(torch::Tensor x) {
          x = torch::relu(fc_input(x));
          x = dropout_input(x);
          x = torch::relu(fc_output(x));
          x = dropout_output(x);
          return x;
        };

        torch::nn::Linear fc_input, fc_output;
        torch::nn::Dropout dropout_input, dropout_output;
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

      int size_value_net = (input_size * hidden_size + hidden_size) + (hidden_size * output_size + output_size);
      int size_policy_net = 96;
};

#endif

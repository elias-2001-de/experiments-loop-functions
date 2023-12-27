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

      virtual void SetTraining(bool value);

    private:
      Real m_fRadius;
      CVector2 m_cCoordBlackSpot;
      CVector2 m_cCoordWhiteSpot;

      Real m_fObjectiveFunction;

      CSpace::TMapPerType tEpuckMap;

      // Get the time step
      int fTimeStep;
      int mission_lengh;
      
      zmq::context_t m_context;
      zmq::socket_t m_socket_actor;
      zmq::socket_t m_socket_critic;

      // Network
      int critic_input_dim;
      int critic_hidden_dim;
      int critic_num_hidden_layers;
      int critic_output_dim;
      int critic_policy_size;

      int actor_input_dim;
      int actor_hidden_dim;
      int actor_num_hidden_layers;
      int actor_output_dim;
      int actor_policy_size;

      int nb_robots;

      torch::Device device = torch::kCPU;

      CRange<Real> m_cNeuralNetworkOutputRange;
      // Define the ConvNet architecture
      struct Net : torch::nn::Module {
        std::vector<torch::nn::Linear> hidden_layers;
        torch::nn::Linear output_layer{nullptr};

        Net(int64_t input_dim = 4, int64_t hidden_dim = 32, int64_t num_hidden_layers = 2, int64_t output_dim = 4) {
            // Create hidden layers
            if(num_hidden_layers > 0){
                for (int64_t i = 0; i < num_hidden_layers; ++i) {
                    int64_t in_features = i == 0 ? input_dim : hidden_dim;
                    hidden_layers.push_back(register_module("fc" + std::to_string(i+1), torch::nn::Linear(in_features, hidden_dim)));
                }

                // Create output layer
                output_layer = register_module("fc" + std::to_string(num_hidden_layers + 1), torch::nn::Linear(hidden_dim, output_dim));
            }else{
                output_layer = register_module("fc", torch::nn::Linear(input_dim, output_dim));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            // Apply hidden layers
            for (auto& layer : hidden_layers) {
                x = torch::relu(layer->forward(x));
            }

            // Apply output layer
            x = output_layer->forward(x);
            return x;
        }

            void print_last_layer_params() {
                // Print parameters of the output layer
                std::cout << "Weights of last layer:\n" << output_layer->weight << std::endl;
                std::cout << "Bias of last layer:\n" << output_layer->bias << std::endl;
            }
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
      torch::Tensor state = torch::empty({critic_input_dim});
      torch::Tensor state_prime = torch::empty({critic_input_dim});
      // torch::Tensor time;
      // torch::Tensor time_prime;

      int size_value_net;
      int size_policy_net;

      float gamma;
      float lambda_critic;

      int64_t port; // TCP interprocess comunication port number for crtic (port+1 for the actor)
};

#endif

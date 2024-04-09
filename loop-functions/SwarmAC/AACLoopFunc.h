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
#include <unordered_map>

#include <iostream>
#include <unistd.h> 
#include <sstream> 

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
      float GetTDError();
      float GetCriticLoss();
      float GetActorLoss();
      float GetEntropy();

      virtual CColor GetFloorColor(const CVector2& c_position_on_plane);

      virtual CVector3 GetRandomPosition();

      double computeBetaLogPDF(double alpha, double beta, double x);

      void GetParametersVector(const torch::nn::Module& module, std::vector<float>& params_vector);

      void print_grid(at::Tensor grid, int step);

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
      
      // Define the Actor and Critic Nets
      struct Critic_Net : torch::nn::Module {
        std::vector<torch::nn::Linear> hidden_layers;
        torch::nn::Linear output_layer{nullptr};

        Critic_Net(int64_t input_dim = 4, int64_t hidden_dim = 64, int64_t num_hidden_layers = 3, int64_t output_dim = 1) {
            // Create hidden layers
            if(num_hidden_layers > 0){
                for (int64_t i = 0; i < num_hidden_layers; ++i) {
                    int64_t in_features = i == 0 ? input_dim : hidden_dim;
                    hidden_layers.push_back(register_module("fc" + std::to_string(i+1), torch::nn::Linear(in_features, hidden_dim)));
                }

                // Create output layer
                output_layer = register_module("fc_output", torch::nn::Linear(hidden_dim, output_dim));
            }else{
                output_layer = register_module("fc", torch::nn::Linear(input_dim, output_dim));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            // Apply hidden layers
            for (auto& layer : hidden_layers) {
                x = torch::tanh(layer->forward(x));
            }

            // Apply output layer
            x = output_layer->forward(x);
            return x;
        }

        void print_last_layer_params() {
            int i = 0;
            for (auto& layer : hidden_layers) {
                std::cout << "Weights of layer " << i << ":\n" << layer->weight << std::endl;
                std::cout << "Weights of layer " << i << ":\n" << layer->bias << std::endl;
                i++;
            }
            // Print parameters of the output layer
            std::cout << "Weights of last layer:\n" << output_layer->weight << std::endl;
            std::cout << "Bias of last layer:\n" << output_layer->bias << std::endl;
        }

        void print_params() {
            int i = 0;
            for (auto& layer : hidden_layers) {
                std::cout << "Weights of layer " << i << ":\n" << layer->weight << std::endl;
                std::cout << "Weights of layer " << i << ":\n" << layer->bias << std::endl;
                i++;
            }
            // Print parameters of the output layer
            std::cout << "Weights of last layer:\n" << output_layer->weight << std::endl;
            std::cout << "Bias of last layer:\n" << output_layer->bias << std::endl;
        }
      };
      Critic_Net critic_net; 
      argos::CEpuckNNController::Actor_Net actor_net;

      // Learning variables
      std::unordered_map<std::string, torch::Tensor> eligibility_trace_critic;
      std::unordered_map<std::string, torch::Tensor> eligibility_trace_actor;
      std::vector<float> value_param;
      std::vector<float> policy_param;
      float I;

      torch::Tensor state = torch::empty({critic_input_dim});
      torch::Tensor state_prime = torch::empty({critic_input_dim});

      int size_value_net;
      int size_policy_net;

      float gamma;
      float lambda_critic;
      float alpha_critic;
      float lambda_actor;
      float alpha_actor;
      float entropy_fact;

      int64_t port; // TCP interprocess comunication port number for crtic (port+1 for the actor)

      std::shared_ptr<torch::optim::Optimizer> optimizer_actor;
      std::shared_ptr<torch::optim::Optimizer> optimizer_critic;

      std::vector<float> TDerrors;
      std::vector<float> Entropies;
      std::vector<float> critic_losses;
      std::vector<float> actor_losses;

};

#endif

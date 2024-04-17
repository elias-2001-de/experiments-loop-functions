/*
 * Aggregation with Ambient Clues
 *
 * @author Antoine Ligot - <aligot@ulb.ac.be>
 */

#ifndef MADDPG_LOOP_FUNC_H
#define MADDPG_LOOP_FUNC_H

#include <torch/torch.h>

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <cmath>

#include "../../src/CoreLoopFunctions.h"
#include "../../../argos3-nn/src/NNController.h"

#include <typeinfo>

#include <random>

using namespace argos;

class MADDPGLoopFunction : public CoreLoopFunctions {
    protected:

    ~MADDPGLoopFunction(){};

    bool training = true;

    int size_value_net;

    int size_policy_net;

  public:
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
                output_layer = register_module("fc" + std::to_string(num_hidden_layers + 1), torch::nn::Linear(hidden_dim, output_dim));
            }else{
                output_layer = register_module("fc", torch::nn::Linear(input_dim, output_dim));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            // Apply hidden layers
            //std::cout << "Nb hidden layers" << hidden_layers << std::endl;
            for (auto& layer : hidden_layers) {
                x = torch::tanh(layer->forward(x));
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

    struct Agent{
        CEPuckEntity* pcEpuck;
        CControllableEntity* pcEntity;
        Critic_Net critic;
        argos::CEpuckNNController::Actor_Net actor;
        Critic_Net target_critic;
        torch::optim::Adam* optimizer_critic;
        torch::optim::Adam* optimizer_actor;
        std::vector<float> policy_param;
        std::vector<float> value_param;

        Agent(int64_t critic_input_dim = 6, int64_t critic_hidden_dim = 64, int64_t critic_num_hidden_layers = 3, int64_t critic_output_dim = 1, int64_t actor_input_dim = 25, int64_t actor_hidden_dim = 64, int64_t actor_num_hidden_layers = 3, 
                int64_t actor_output_dim = 2, float lambda_actor = 0.9, float lambda_critic = 0.8, int size_value_net = 5, int size_policy_net = 52){
            //std::cout << "Before creation of a new agent" << "." << std::endl;
            pcEpuck = nullptr;
            pcEntity = nullptr;
            //std::cout << "Critic input dim" << critic_input_dim << std::endl;
            critic = Critic_Net(critic_input_dim, critic_hidden_dim, critic_num_hidden_layers, critic_output_dim);
            actor = argos::CEpuckNNController::Actor_Net(actor_input_dim, actor_hidden_dim, actor_num_hidden_layers, actor_output_dim);
            optimizer_actor = new torch::optim::Adam(actor.parameters(), torch::optim::AdamOptions(lambda_actor));
            optimizer_actor->zero_grad();
            optimizer_critic = new torch::optim::Adam(critic.parameters(), torch::optim::AdamOptions(lambda_critic));
            optimizer_critic->zero_grad();
            policy_param.resize(size_policy_net);
            std::fill(policy_param.begin(), policy_param.end(), 0.0f);
            value_param.resize(size_value_net);
            std::fill(value_param.begin(), value_param.end(), 0.0f);

            //Initialisation of the parameters of the networks
            size_t index = 0;
            for (auto& param : critic.parameters()) {
	            auto param_data = torch::from_blob(value_param.data() + index, param.data().sizes()).clone();
	            param.data() = param_data;
                index += param_data.numel();
            }
            index = 0;
            for (auto& param : actor.parameters()) {
	            auto param_data = torch::from_blob(policy_param.data() + index, param.data().sizes()).clone();
	            param.data() = param_data;
                index += param_data.numel();
            }
            target_critic = critic;
            //std::cout << "After creation of a new agent" << "." << std::endl;
        }
      };

      //Buffer
      struct Transition{
        torch::Tensor state;
        std::vector<torch::Tensor> obs;
        std::vector<torch::Tensor> actions;
        std::vector<int> rewards;
        torch::Tensor state_prime;
        std::vector<torch::Tensor> next_obs;

        Transition(torch::Tensor cstr_state, std::vector<torch::Tensor> cstr_obs, std::vector<torch::Tensor> cstr_actions, std::vector<int> cstr_rewards, torch::Tensor cstr_state_prime, std::vector<torch::Tensor> cstr_next_obs) {
            state = cstr_state;
            obs = cstr_obs;
            actions = cstr_actions;
            rewards = cstr_rewards;
            state_prime = cstr_state_prime;
            next_obs = cstr_next_obs;
        }
      };

      virtual void SetVectorAgents(std::vector<Agent*> vectorAgents) {}

      virtual void SetBuffer(std::vector<Transition*> vectorTransitions) {}

      virtual void SetTraining(bool value) {}

      virtual void SetSizeValueNet(int size_value_net) {}

      virtual void SetSizePolicyNet(int size_policy_net) {}

      virtual void SetControllerEpuckAgent() {}

};
#endif

/*
 * @file <src/CoreLoopFunctions.h>
 *
 * @author Antoine Ligot - <aligot@ulb.ac.be>
 *
 * @package ARGoS3-AutoMoDe
 *
 * @license MIT License
 */

#ifndef AUTOMODE_LOOP_FUNCTIONS
#define AUTOMODE_LOOP_FUNCTIONS

#include <torch/torch.h>

#include <argos3/core/simulator/space/space.h>
#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

// FIX not very clean
#include "../../BC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

using namespace argos;

class CoreLoopFunctions: public CLoopFunctions {
  protected:

    ~CoreLoopFunctions();

    /*
     * Reset function. First remove the robots, then re-position them.
     */
    virtual void Reset();

    /*
     * Method used to remove the robots from the arena.
     */
    void RemoveRobots();

    /*
     * Method used to reallocate the robots.
     * The position is given by the method GetRandomPosition().
     */
    void MoveRobots();

    /*
     * The number of robots to be placed for the experiment.
     */
    UInt32 m_unNumberRobots;

    /*
     * The radius of the circle in which the robots will be distributed.
     */
    Real m_fDistributionRadius;

    CRandom::CRNG* m_pcRng;

  public:
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

    /*
     * The pointers to the shared memory containing the policy and the value
     */
    Critic_Net* critic_net;
    argos::CEpuckNNController::ActorBase* actor_net;

    bool training;
    /*
     * Initialization method where the parameters of the loop function
     * are fetched from the xml declaration.
     */
    virtual void Init(argos::TConfigurationNode& t_tree);

    /*
     * This method should return the fitness of the controller.
     */
    virtual Real GetObjectiveFunction() = 0;

    virtual float GetTDError() = 0;
    virtual float GetCriticLoss() = 0;
    virtual float GetActorLoss() = 0;
    virtual float GetEntropy() = 0;
    virtual std::vector<float> GetBehavHist() = 0;

    /*
     * Return a random position.
     */
    virtual CVector3 GetRandomPosition() = 0;

    /*
     * Returns the Behavioral characterization vector, for repertoire generation
     */
    //virtual std::vector<Real> GetSDBC() {}; 

    /*
     * Sets the policy pointer
     */
    virtual void SetPolicyPointer(argos::CEpuckNNController::ActorBase* policy);

    /*
     * Sets the value pointer
     */ 
    virtual void SetValuePointer(Critic_Net* value);

    virtual void SetTraining(bool value);
};

#endif

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

#include <cmath>
#include <unordered_map>

#include <iostream>
#include <unistd.h> 
#include <sstream> 

#include "../../src/CoreLoopFunctions.h"
#include "../../../BC_trainer/src/shared_mem.h"
#include "../../../argos3-nn/src/NNController.h"

#include <typeinfo>

struct RelativePosition {
    float distance;
    float angle;
    float relative_yaw;

    // Default constructor
    RelativePosition() : distance(0), angle(0), relative_yaw(0) {}

    // Constructor with parameters
    RelativePosition(float d, float a, float ry) : distance(d), angle(a), relative_yaw(ry) {}
};

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
      std::vector<float> GetBehavHist();

      std::vector<RelativePosition> compute_relative_positions(const CVector2& base_position, const CRadians& base_yaw, const std::vector<std::pair<CVector2, CRadians>>& all_positions);

      virtual CColor GetFloorColor(const CVector2& c_position_on_plane);

      virtual CVector3 GetRandomPosition();

      virtual void SetTraining(bool value);

    private:
      Real m_fRadius;
      CVector2 m_cCoordBlackSpot;
      CVector2 m_cCoordWhiteSpot;

      Real m_fObjectiveFunction;

      CSpace::TMapPerType tEpuckMap;

      // Get the time step
      int fTimeStep;
      int mission_length;

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
      int n_sensors;

      std::string actor_type; 
      std::string device_type = "cuda"; 
      torch::Device device = torch::kCUDA;
    
      CRange<Real> m_cNeuralNetworkOutputRange;
      
      // Learning variables
      std::unordered_map<std::string, torch::Tensor> eligibility_trace_critic;
      std::unordered_map<std::string, torch::Tensor> eligibility_trace_behavior;
      std::unordered_map<std::string, torch::Tensor> eligibility_trace_terminator;
      std::vector<float> value_param;
      std::vector<float> behavior_param;
      std::vector<float> terminator_param;

      std::vector<torch::Tensor> states;
      std::vector<torch::Tensor> states_prime;
      std::vector<torch::Tensor> observations;

      int size_value_net;
      int size_behavior_net;
      int size_terminator_net;

      float gamma;
      float lambda_critic;
      float alpha_critic;
      float lambda_behavior;
      float alpha_behavior;
      float lambda_terminator;
      float alpha_terminator;
      float entropy_fact;

      std::shared_ptr<torch::optim::Optimizer> optimizer_behavior;
      std::shared_ptr<torch::optim::Optimizer> optimizer_terminator;
      std::shared_ptr<torch::optim::Optimizer> optimizer_critic;

      std::vector<torch::Tensor> all_module_probabilities;
      std::vector<torch::Tensor> termination_probabilities;
      std::vector<int> selected_modules;

      std::vector<float> TDerrors;
      std::vector<float> Entropies;
      std::vector<float> critic_losses;
      std::vector<float> actor_losses;
      std::vector<float> behav_hist;
};

#endif

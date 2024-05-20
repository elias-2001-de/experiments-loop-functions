/*
 * Aggregation with Ambient Clues
 *
 * @author Antoine Ligot - <aligot@ulb.ac.be>
 */

#ifndef CHOCOLATE_AAC_LOOP_FUNC_H
#define CHOCOLATE_AAC_LOOP_FUNC_H

#include <torch/torch.h>
#include <fstream>

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include <cmath>

#include "../MADDPGLoopFunc.h"
#include "../../../../argos3-nn/src/NNController.h"

#include <typeinfo>

#include <random>
#include <ctime>

using namespace argos;

class AACLoopFunction : public MADDPGLoopFunction {

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

      void RemoveArena();

      Real GetObjectiveFunction();

      float GetCriticLoss();

      float GetActorLoss();

      float GetEntropy();

      virtual CColor GetFloorColor(const CVector2& c_position_on_plane);

      virtual CVector3 GetRandomPosition();

      double computeBetaLogPDF(double alpha, double beta, double x);

      void update_actor(torch::nn::Module actor_net, torch::Tensor& behavior_probs, 
                  torch::Tensor& beta_parameters, int chosen_behavior, 
                  double chosen_parameter, double td_error, 
                  torch::optim::Adam* optimizer);

      void print_grid(at::Tensor grid);

      virtual void SetTraining(bool value);
      
      virtual void SetVectorAgents(std::vector<MADDPGLoopFunction::Agent*> vectorAgents);

      virtual void SetSizeValueNet(int value);

      virtual void SetSizePolicyNet(int policy);

      virtual void SetControllerEpuckAgent();

    private:
      Real m_fRadius;
      CVector2 m_cCoordBlackSpot;
      CVector2 m_cCoordWhiteSpot;

      Real m_fObjectiveFunction;

      CSpace::TMapPerType tEpuckMap;

      int nb_robots;

      // Get the time step
      int fTimeStep;
      int mission_lengh;
      int fTimeStepTraining;

      // Networks
      int critic_input_dim;

      int actor_input_dim;

      float gamma;
      float tau;
      int batch_size;
      int batch_begin;
      int batch_step;
      int update_target;
      int max_buffer_size;

      int size_value_net;

      int size_policy_net;

      bool training = true;
      int fTimeBatch;
      int fEpisode = 0;

      std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();;

      torch::Device device = torch::kCPU;

      CRange<Real> m_cNeuralNetworkOutputRange;

      std::vector<MADDPGLoopFunction::Agent*> agents;

      CircularBuffer<MADDPGLoopFunction::Transition*> buffer;

      std::vector<torch::Tensor> obs;

      torch::Tensor state;
      torch::Tensor state_prime;

      float critic_losses = 0.0;
      float actor_losses = 0.0;
      float entropies = 0.0;
};
#endif
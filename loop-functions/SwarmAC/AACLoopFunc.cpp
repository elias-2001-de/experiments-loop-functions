/**
  * @file <loop-functions/example/ChocolateAACLoopFunc.cpp>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

#include "AACLoopFunc.h"
#include <cuda_runtime.h>
#include <chrono>

/****************************************/
/****************************************/

AACLoopFunction::AACLoopFunction() {
  m_fRadius = 0.3;
  m_cCoordBlackSpot = CVector2(0,-0.6);
  m_cCoordWhiteSpot = CVector2(0,0.6);
}

/****************************************/
/****************************************/

AACLoopFunction::AACLoopFunction(const AACLoopFunction& orig) {}

/****************************************/
/****************************************/

AACLoopFunction::~AACLoopFunction() {}

/****************************************/
/****************************************/

void AACLoopFunction::Destroy() {}

/****************************************/
/****************************************/

void AACLoopFunction::Reset() {
  m_fObjectiveFunction = 0;
  fTimeStep = 0;

  // Clear existing eligibility traces and initialize them on the device
  eligibility_trace_critic.clear();
  eligibility_trace_actor.clear();

  for (const auto& named_param : critic_net->named_parameters()) {
      if (named_param.value().requires_grad()) {
          try {
              eligibility_trace_critic[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
          } catch (const std::exception& ex) {
              std::cerr << "Error while initializing eligibility_trace_critic for " << named_param.key() << ": " << ex.what() << std::endl;
          }
      }
  }

  for (const auto& named_param : actor_net->named_parameters()) {
    if (named_param.value().requires_grad()) {
        try {
            eligibility_trace_actor[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
        } catch (const std::exception& ex) {
            std::cerr << "Error while initializing eligibility_trace_actor for " << named_param.key() << ": " << ex.what() << std::endl;
        }
    }
  }

  optimizer_actor = std::make_shared<torch::optim::Adam>(actor_net->parameters(), torch::optim::AdamOptions(alpha_critic));
  optimizer_critic = std::make_shared<torch::optim::Adam>(critic_net->parameters(), torch::optim::AdamOptions(alpha_actor));
  optimizer_critic->zero_grad();
  optimizer_actor->zero_grad();
  //std::cout << "[loop] actor device: " << actor_net->parameters().front().device() << std::endl;;
  //std::cout << "[loop] critic device: " << critic_net->parameters().front().device() << std::endl;;

  I = 1;

  TDerrors.resize(mission_length);
  std::fill(TDerrors.begin(), TDerrors.end(), 0.0f);

  Entropies.resize(mission_length);
  std::fill(Entropies.begin(), Entropies.end(), 0.0f);

  critic_losses.resize(mission_length);
  std::fill(critic_losses.begin(), critic_losses.end(), 0.0f);

  actor_losses.resize(mission_length);
  std::fill(actor_losses.begin(), actor_losses.end(), 0.0f);

  behav_hist.resize(actor_output_dim);
  std::fill(behav_hist.begin(), behav_hist.end(), 0.0f);

  CoreLoopFunctions::Reset();
}

/****************************************/
/****************************************/

void AACLoopFunction::Init(TConfigurationNode& t_tree) {
  CoreLoopFunctions::Init(t_tree);
  TConfigurationNode cParametersNode;
  cParametersNode = GetNode(t_tree, "params");

  GetNodeAttribute(cParametersNode, "number_robots", nb_robots);
  GetNodeAttribute(cParametersNode, "actor_type", actor_type);

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", critic_input_dim);
  GetNodeAttribute(criticParameters, "hidden_dim", critic_hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", critic_num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", critic_output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);
  GetNodeAttribute(criticParameters, "alpha_critic", alpha_critic);
  GetNodeAttribute(criticParameters, "gamma", gamma);
  GetNodeAttribute(criticParameters, "device", device_type);
  device = (device_type == "cuda") ? torch::kCUDA : torch::kCPU;

  TConfigurationNode actorParameters;
  argos::TConfigurationNode& parentNode = *dynamic_cast<argos::TConfigurationNode*>(t_tree.Parent());
  TConfigurationNode controllerNode;
  TConfigurationNode actorNode;
  controllerNode = GetNode(parentNode, "controllers");
  actorNode = GetNode(controllerNode, actor_type + "_controller");
  actorParameters = GetNode(actorNode, "actor");
  GetNodeAttribute(actorParameters, "input_dim", actor_input_dim);
  GetNodeAttribute(actorParameters, "hidden_dim", actor_hidden_dim);
  GetNodeAttribute(actorParameters, "num_hidden_layers", actor_num_hidden_layers);
  GetNodeAttribute(actorParameters, "output_dim", actor_output_dim);
  GetNodeAttribute(actorParameters, "lambda_actor", lambda_actor);
  GetNodeAttribute(actorParameters, "alpha_actor", alpha_actor);
  GetNodeAttribute(actorParameters, "entropy", entropy_fact);

  if (actor_type == "dandel") {
    actor_net = dynamic_cast<argos::CEpuckNNController::Dandel*>(actor_net);
    n_sensors = 16;
  }else if (actor_type == "daisy"){
    actor_net = dynamic_cast<argos::CEpuckNNController::Daisy*>(actor_net);
    n_sensors = 22;
  }

  TConfigurationNode frameworkNode;
  TConfigurationNode experimentNode;
  frameworkNode = GetNode(parentNode, "framework");
  experimentNode = GetNode(frameworkNode, "experiment");
  GetNodeAttribute(experimentNode, "length", mission_length);
  mission_length *= 10;
  observations.resize(nb_robots, torch::zeros({n_sensors}, device));
  all_module_probabilities.resize(nb_robots, torch::zeros({6}, device));
  selected_modules.resize(nb_robots);

  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
  for (auto it = cEntities.begin(); it != cEntities.end(); ++it) {
    CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
    try {
        CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
        cController.SetDevice(device);
    } catch (std::exception &ex) {
        LOGERR << "Error while setting device: " << ex.what() << std::endl;
    }
  }
}

/****************************************/
/****************************************/

argos::CColor AACLoopFunction::GetFloorColor(const argos::CVector2& c_position_on_plane) {
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  Real d = (m_cCoordBlackSpot - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::BLACK;
  }

  d = (m_cCoordWhiteSpot - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::WHITE;
  }

  return CColor::GRAY50;
}

/****************************************/
/****************************************/

void AACLoopFunction::PreStep() {
  if (fTimeStep == 0)
  {
    std::vector<torch::Tensor> positions;
    CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0,0), other_position(0,0);
    CRadians cRoll, cPitch, cYaw, other_yaw;
    std::vector<torch::Tensor> rewards;
    int N = (critic_input_dim - 4)/5; // Number of closest neighbors to consider (4 absolute states + 5 relative states)

    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
      CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
      cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                        pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                        
      pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation.ToEulerAngles(cYaw, cPitch, cRoll);

      // Collect positions and orientations of other e-pucks
      std::vector<std::pair<CVector2, CRadians>> all_positions;
      CQuaternion cEpuckOrientation;
      for (CSpace::TMapPerType::iterator jt = tEpuckMap.begin(); jt != tEpuckMap.end(); ++jt) {
        if (jt != it) {
          CEPuckEntity* pcOtherEpuck = any_cast<CEPuckEntity*>(jt->second);
          other_position.Set(pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                            pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

          pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation.ToEulerAngles(other_yaw, cPitch, cRoll);
          all_positions.emplace_back(other_position, other_yaw);
        }
      }

      // Compute relative positions and orientations
      std::vector<RelativePosition> relatives = compute_relative_positions(cEpuckPosition, cYaw, all_positions);

      // Sort by distance and pick the N closest
      std::nth_element(relatives.begin(), relatives.begin() + N, relatives.end(), [](const RelativePosition& a, const RelativePosition& b) {
        return a.distance < b.distance;
      });

      // Create state tensor for the current e-puck
      torch::Tensor pos = torch::empty({critic_input_dim}, device);
      pos[0] = cEpuckPosition.GetX();
      pos[1] = cEpuckPosition.GetY();
      pos[2] = std::cos(cYaw.GetValue());
      pos[3] = std::sin(cYaw.GetValue());
      int index = 4;

      // Iterate only over the first N elements of relatives
      for (int i = 0; i < N; ++i) {
        const auto& rel = relatives[i];
        pos[index++] = rel.distance;
        pos[index++] = std::cos(rel.angle);
        pos[index++] = std::sin(rel.angle);
        pos[index++] = std::cos(rel.relative_yaw);
        pos[index++] = std::sin(rel.relative_yaw);
      }

      positions.push_back(pos);

      // Reward computation
      float reward = 0.0;
      Real fDistanceSpot = (m_cCoordBlackSpot - cEpuckPosition).Length();
      if (fDistanceSpot <= m_fRadius) {
        reward = 1.0;
      } else {
        reward = 0; //std::pow(10, -3 * fDistanceSpot / m_fDistributionRadius);
      }
      rewards.push_back(torch::tensor(reward));
      m_fObjectiveFunction += reward;
    }
    states = positions;
  }

  // Launch the experiment with the correct random seed and network,
  // and evaluate the average fitness

  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
  int i = 0;

  /// ACTOR DECISION
  i = 0;
  for (auto it = cEntities.begin(); it != cEntities.end(); ++it) {
    CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
    try {
        CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());

        if (actor_type == "daisy") {
            // Use GPU tensor for forward pass
            //auto start = std::chrono::high_resolution_clock::now();
            observations[i] = cController.GetObservations();
            // auto end = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            // std::cout << "Execution time GetObservations: " << duration.count() << " microseconds" << std::endl;

            //start = std::chrono::high_resolution_clock::now();
            torch::Tensor module_probabilities = dynamic_cast<argos::CEpuckNNController::Daisy*>(actor_net)->forward(observations[i]);
            //end = std::chrono::high_resolution_clock::now();
            //duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            //std::cout << "Execution time forward: " << duration.count() << " microseconds" << std::endl;

            all_module_probabilities[i] = module_probabilities;
            selected_modules[i] = module_probabilities.multinomial(1).item<int>();
            cController.SetSelectedModule(selected_modules[i]);
        }
        i++;
    } catch (std::exception &ex) {
        LOGERR << "Error while setting network: " << ex.what() << std::endl;
    }
  }
}


/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  if(training){
    std::vector<torch::Tensor> positions;
    CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0,0), other_position(0,0);
    CRadians cRoll, cPitch, cYaw, other_yaw;
    std::vector<torch::Tensor> rewards;
    int N = (critic_input_dim - 4)/5; // Number of closest neighbors to consider (4 absolute states + 5 relative states)

    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
      CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
      cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                        pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                        
      pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation.ToEulerAngles(cYaw, cPitch, cRoll);

      // Collect positions and orientations of other e-pucks
      std::vector<std::pair<CVector2, CRadians>> all_positions;
      CQuaternion cEpuckOrientation;
      for (CSpace::TMapPerType::iterator jt = tEpuckMap.begin(); jt != tEpuckMap.end(); ++jt) {
        if (jt != it) {
          CEPuckEntity* pcOtherEpuck = any_cast<CEPuckEntity*>(jt->second);
          other_position.Set(pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                            pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

          pcOtherEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation.ToEulerAngles(other_yaw, cPitch, cRoll);
          all_positions.emplace_back(other_position, other_yaw);
        }
      }

      // Compute relative positions and orientations
      std::vector<RelativePosition> relatives = compute_relative_positions(cEpuckPosition, cYaw, all_positions);

      // Sort by distance and pick the N closest
      std::nth_element(relatives.begin(), relatives.begin() + N, relatives.end(), [](const RelativePosition& a, const RelativePosition& b) {
        return a.distance < b.distance;
      });

      // Create state tensor for the current e-puck
      torch::Tensor pos = torch::empty({critic_input_dim}, torch::kCPU);
      pos[0] = cEpuckPosition.GetX();
      pos[1] = cEpuckPosition.GetY();
      pos[2] = std::cos(cYaw.GetValue());
      pos[3] = std::sin(cYaw.GetValue());
      int index = 4;

      // Iterate only over the first N elements of relatives
      for (int i = 0; i < N; ++i) {
        const auto& rel = relatives[i];
        pos[index++] = rel.distance;
        pos[index++] = std::cos(rel.angle);
        pos[index++] = std::sin(rel.angle);
        pos[index++] = std::cos(rel.relative_yaw);
        pos[index++] = std::sin(rel.relative_yaw);
      }

      positions.push_back(pos);

      // Reward computation
      float reward = 0.0;
      Real fDistanceSpot = (m_cCoordBlackSpot - cEpuckPosition).Length();
      if (fDistanceSpot <= m_fRadius) {
        reward = 1.0;
      } else {
        reward = 0; //std::pow(10, -3 * fDistanceSpot / m_fDistributionRadius);
      }
      rewards.push_back(torch::tensor(reward));
      m_fObjectiveFunction += reward;
    }

    try{
      states_prime = positions;

      torch::Tensor state_batch = torch::stack(states).to(device);
      torch::Tensor next_state_batch = torch::stack(states_prime).to(device);
      torch::Tensor reward_batch = torch::stack(rewards).unsqueeze(1).to(device);

      torch::Tensor v_state = critic_net->forward(state_batch);
      torch::Tensor v_state_prime = critic_net->forward(next_state_batch);

      if (fTimeStep + 1 == mission_length) {
        v_state_prime = torch::zeros_like(v_state_prime);
      }

      torch::Tensor td_errors = reward_batch + gamma * v_state_prime - v_state;
      TDerrors[fTimeStep] = td_errors.mean().item<float>();

      torch::Tensor critic_loss = td_errors.pow(2).mean();
      critic_losses[fTimeStep] = critic_loss.item<float>();

      optimizer_critic->zero_grad();
      critic_loss.backward();
      for (auto& named_param : critic_net->named_parameters()) {
        if (named_param.value().grad().defined()) {
          eligibility_trace_critic[named_param.key()].mul_(gamma * lambda_critic).add_(named_param.value().grad());
          named_param.value().mutable_grad() = eligibility_trace_critic[named_param.key()];
        }
      }
      optimizer_critic->step();

      // Policy update
      std::vector<torch::Tensor> log_probs, entropies;
      CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");

      int i = 0;
      for (auto& it : cEntities) {
        CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(any_cast<CControllableEntity*>(it.second)->GetController());
        log_probs.push_back(torch::log(all_module_probabilities[i][selected_modules[i]]));
        entropies.push_back(-torch::sum(all_module_probabilities[i] * torch::log(all_module_probabilities[i] + 1e-20)));
        behav_hist[selected_modules[i]] += 1;
        i++;
      }

      torch::Tensor log_probs_batch = torch::stack(log_probs).unsqueeze(1);
      torch::Tensor policy_loss = -(log_probs_batch * td_errors.detach()).mean();

      torch::Tensor entropy_batch = torch::stack(entropies).unsqueeze(1);
      policy_loss -= entropy_fact * entropy_batch.mean();

      actor_losses[fTimeStep] = policy_loss.item<float>();
      Entropies[fTimeStep] = entropy_batch.mean().item<float>();
      optimizer_actor->zero_grad();
      policy_loss.backward();
      for (auto& named_param : actor_net->named_parameters()) {
        if (named_param.value().grad().defined()) {
          eligibility_trace_actor[named_param.key()].mul_(gamma * lambda_actor).add_(named_param.value().grad() * I);
          named_param.value().mutable_grad() = eligibility_trace_actor[named_param.key()];
        }
      }
      optimizer_actor->step();

      I *= gamma;
    }catch (std::exception &ex) {
      LOGERR << "Error in training loop: " << ex.what() << std::endl;
    }

    states = states_prime;
  }
  fTimeStep += 1;
}

/****************************************/
/****************************************/

void AACLoopFunction::PostExperiment() {
  LOG << "Time = " << fTimeStep << std::endl;
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
}

/****************************************/
/****************************************/

Real AACLoopFunction::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

float AACLoopFunction::GetTDError() {
  float sum = std::accumulate(TDerrors.begin(), TDerrors.end(), 0.0f);
  float average = sum / TDerrors.size();
  return average;
}

/****************************************/
/****************************************/

float AACLoopFunction::GetEntropy() {
  float sum = std::accumulate(Entropies.begin(), Entropies.end(), 0.0f);
  float average = sum / Entropies.size();
  return average;
}

/****************************************/
/****************************************/

float AACLoopFunction::GetActorLoss() {
  float sum = std::accumulate(actor_losses.begin(), actor_losses.end(), 0.0f);
  float average = sum / actor_losses.size();
  return average;
}


/****************************************/
/****************************************/

float AACLoopFunction::GetCriticLoss() {
  float sum = std::accumulate(critic_losses.begin(), critic_losses.end(), 0.0f);
  float average = sum / critic_losses.size();
  return average;
}

/****************************************/
/****************************************/

std::vector<float> AACLoopFunction::GetBehavHist() {
  //Calculate the sum of all elements
  float sum = std::accumulate(behav_hist.begin(), behav_hist.end(), 0.0f);

  //Convert each element to percentage
  if (sum != 0.0f) { 
      for (auto& elem : behav_hist) {
          elem = (elem / sum) * 100.0f;
      }
  }

  return behav_hist;
}

/****************************************/
/****************************************/

std::vector<RelativePosition> AACLoopFunction::compute_relative_positions(const CVector2& base_position, const CRadians& base_yaw, const std::vector<std::pair<CVector2, CRadians>>& all_positions) {
  std::vector<RelativePosition> relative_positions;
  for (const auto& pos : all_positions) {
    float dx = pos.first.GetX() - base_position.GetX();
    float dy = pos.first.GetY() - base_position.GetY();
    float distance = std::sqrt(dx * dx + dy * dy);
    float angle = std::atan2(dy, dx) - base_yaw.GetValue();
    float relative_yaw = pos.second.GetValue() - base_yaw.GetValue();

    relative_positions.emplace_back(distance, angle, relative_yaw);
  }
  return relative_positions;
}

/****************************************/
/****************************************/

CVector3 AACLoopFunction::GetRandomPosition() {
  Real temp;
  Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  Real  b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  // If b < a, swap them
  if (b < a) {
    temp = a;
    a = b;
    b = temp;
  }
  Real fPosX = b * m_fDistributionRadius * cos(2 * CRadians::PI.GetValue() * (a/b));
  Real fPosY = b * m_fDistributionRadius * sin(2 * CRadians::PI.GetValue() * (a/b));

  return CVector3(fPosX, fPosY, 0);
}

/****************************************/
/****************************************/

void AACLoopFunction::SetTraining(bool value) {
    training = value;
}

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");

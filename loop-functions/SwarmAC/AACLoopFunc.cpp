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

/****************************************/
/****************************************/

AACLoopFunction::AACLoopFunction() {
  m_fRadius = 0.3;
  m_cCoordBlackSpot = CVector2(0,-0.6);
  m_cCoordWhiteSpot = CVector2(0,0.6);
  m_fObjectiveFunction = 0;
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
  //std::cout << "RESET\n";
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
  //std::cout << "INIT\n";
  CoreLoopFunctions::Init(t_tree);
  TConfigurationNode cParametersNode;
  cParametersNode = GetNode(t_tree, "params");

  GetNodeAttribute(cParametersNode, "number_robots", nb_robots);
  GetNodeAttribute(cParametersNode, "actor_type", actor_type);

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", critic_input_dim);
  // critic_input_dim *= nb_robots;
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

  TConfigurationNode frameworkNode;
  TConfigurationNode experimentNode;
  frameworkNode = GetNode(parentNode, "framework");
  experimentNode = GetNode(frameworkNode, "experiment");
  GetNodeAttribute(experimentNode, "length", mission_length);
  mission_length *= 10;
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
  //std::cout << "Presetp\n";
  int N = (critic_input_dim - 4)/5; // Number of closest neighbors to consider (4 absolute states + 5 relative states)
  at::Tensor grid = torch::zeros({50, 50});
  std::vector<torch::Tensor> positions;
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0), other_position(0,0);
  CRadians cRoll, cPitch, cYaw, other_yaw;
  std::vector<torch::Tensor> rewards;

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
    relatives.resize(N);  // Keep only the N closest

    // Create state tensor for the current e-puck
    torch::Tensor pos = torch::empty({critic_input_dim});
    pos[0] = cEpuckPosition.GetX();
    pos[1] = cEpuckPosition.GetY();
    pos[2] = std::cos(cYaw.GetValue());
    pos[3] = std::sin(cYaw.GetValue());
    int index = 4;
    for (const auto& rel : relatives) {
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
      reward = std::pow(10, -3 * fDistanceSpot / m_fDistributionRadius);
    }
    rewards.push_back(torch::tensor(reward));
    m_fObjectiveFunction += reward;

    int grid_x = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    grid[grid_x][grid_y] = 1;
  }
  
  if (fTimeStep == 0)
  {
    states.clear();
    for (const auto& state : positions)
    {
      states.push_back(state.clone());
    }
    
    // Launch the experiment with the correct random seed and network,
    // and evaluate the average fitness
    CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
    int i = 0;
    for (CSpace::TMapPerType::iterator it = cEntities.begin();
        it != cEntities.end(); ++it) {
      CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
      try {
        CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
        cController.SetNetworkAndOptimizer(actor_net, optimizer_actor, device_type);
        // std::cout << "state " << i << " : " <<  states[i] << std::endl;
        // cController.SetGlobalState(states[i]);
        i++;
      } catch (std::exception &ex) {
        LOGERR << "Error while setting network: " << ex.what() << std::endl;
      }
    }

    fTimeStep += 1;
    // std::cout << "Step: " << fTimeStep << std::endl;
  
  }else
  {
    if (training) {
      try{
        states_prime.clear();
        for (const auto& state : positions)
        {
          states_prime.push_back(state.clone());
        }
        torch::Tensor state_batch = torch::stack(states).to(device);
        torch::Tensor next_state_batch = torch::stack(states_prime).to(device);

        // Forward passes
        torch::Tensor v_state = critic_net->forward(state_batch);
        torch::Tensor v_state_prime = critic_net->forward(next_state_batch);
        

        // Adjust for the terminal state
        if (fTimeStep + 1 == mission_length) {
            v_state_prime = torch::zeros_like(v_state_prime);
        }

        // TD error
        torch::Tensor reward_batch = torch::stack(rewards).unsqueeze(1).to(device);
        torch::Tensor td_errors = reward_batch + gamma * v_state_prime - v_state;
        
        TDerrors[fTimeStep] = td_errors.mean().item<float>();

        // Critic Loss
        torch::Tensor critic_loss = td_errors.pow(2).mean();
        critic_losses[fTimeStep] = critic_loss.item<float>();

        // Backward and optimize for critic
        optimizer_critic->zero_grad();
        critic_loss.backward();
        // Update eligibility traces and gradients
        for (auto& named_param : critic_net->named_parameters()) {
            auto& param = named_param.value();
            if (param.grad().defined()) {
                auto& eligibility = eligibility_trace_critic[named_param.key()];
                eligibility.mul_(gamma * lambda_critic).add_(param.grad());
                param.mutable_grad() = eligibility.clone();
            }
        }
        optimizer_critic->step();

        // Prepare for policy update
        std::vector<torch::Tensor> log_probs;
        std::vector<torch::Tensor> entropies;
        CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller"); 
        for (CSpace::TMapPerType::iterator it = cEntities.begin();
            it != cEntities.end(); ++it) {
          CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);      
          CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
          log_probs.push_back(cController.GetPolicyLogProbs().to(device)); 
          entropies.push_back(cController.GetPolicyEntropy().to(device)); 
          int selected_module = cController.GetSelectedModule();
          behav_hist[selected_module] += 1;
        }

        // Calculate policy loss using the per-robot TD error times the logprob
        torch::Tensor log_probs_batch = torch::stack(log_probs).unsqueeze(1);
        torch::Tensor policy_loss = -(log_probs_batch * td_errors.detach()).mean();

        // Add entropy term for better exploration
        torch::Tensor entropy_batch = torch::stack(entropies).unsqueeze(1);
        policy_loss -= entropy_fact * entropy_batch.mean();

        actor_losses[fTimeStep] = policy_loss.item<float>();
        Entropies[fTimeStep] = entropy_batch.mean().item<float>();

        // Backward and optimize for actor
        optimizer_actor->zero_grad();
        policy_loss.backward();
        // Update eligibility traces and gradients
        for (auto& named_param : actor_net->named_parameters()) {
            auto& param = named_param.value();
            if (param.grad().defined()) {
                auto& eligibility = eligibility_trace_actor[named_param.key()];
                eligibility.mul_(gamma * lambda_actor).add_(param.grad() * I);
                param.mutable_grad() = eligibility.clone();
            }
        }
        optimizer_actor->step();

        I *= gamma; // Update the decay term for the eligibility traces
      }catch (std::exception &ex) {
        LOGERR << "Error in training loop: " << ex.what() << std::endl;
      }
    }
    
    states.clear();
    for (const auto& state : positions)
    {
      states.push_back(state.clone());
    }
    
    // Launch the experiment with the correct random seed and network,
    // and evaluate the average fitness
    CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
    int i = 0;
    for (CSpace::TMapPerType::iterator it = cEntities.begin();
        it != cEntities.end(); ++it) {
      CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
      try {
        CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
        cController.SetNetworkAndOptimizer(actor_net, optimizer_actor, device_type);
        i++;
      } catch (std::exception &ex) {
        LOGERR << "Error while setting network: " << ex.what() << std::endl;
      }
    }

    fTimeStep += 1;
  }
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  // if(!training){
  //   // place spot in grid
  //   int grid_x = static_cast<int>(std::round((m_cCoordBlackSpot.GetX() + 1.231) / 2.462 * 49));
  //   int grid_y = static_cast<int>(std::round((-m_cCoordBlackSpot.GetY() + 1.231) / 2.462 * 49));
  //   int radius = static_cast<int>(std::round(m_fRadius / 2.462 * 49));
  //   // Print arena on the terminal
  //   auto temp1 = grid[grid_x][grid_y];
  //   auto temp2 = grid[grid_x+radius][grid_y];
  //   auto temp3 = grid[grid_x-radius][grid_y];
  //   auto temp4 = grid[grid_x][grid_y+radius];
  //   auto temp5 = grid[grid_x][grid_y-radius];
  //   grid[grid_x][grid_y] = 2;
  //   grid[grid_x+radius][grid_y] = 3;
  //   grid[grid_x-radius][grid_y] = 3;
  //   grid[grid_x][grid_y+radius] = 3;
  //   grid[grid_x][grid_y-radius] = 3;
  //   print_grid(grid, fTimeStep);
  //   grid[grid_x][grid_y] = temp1;
  //   grid[grid_x+radius][grid_y] = temp2;
  //   grid[grid_x-radius][grid_y] = temp3;
  //   grid[grid_x][grid_y+radius] = temp4;
  //   grid[grid_x][grid_y-radius] = temp5;
  // }
  // std::cout << "Step: " << fTimeStep << std::endl;
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

// Function to compute the log-PDF of the Beta distribution
double AACLoopFunction::computeBetaLogPDF(double alpha, double beta, double x) {
    // Compute the logarithm of the Beta function
    double logBeta = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);

    // Compute the log-PDF of the Beta distribution
    double logPDF = (alpha - 1) * std::log(x) + (beta - 1) * std::log(1 - x) - logBeta;

    return logPDF;
}

/****************************************/
/****************************************/

void AACLoopFunction::print_grid(at::Tensor grid, int step) {
    std::stringstream buffer;
    buffer << "\033[2J\033[1;1H"; // Clear screen and move cursor to top-left
    buffer << "GRID State:\n";
    
    auto sizes = grid.sizes();
    int rows = sizes[0];
    int cols = sizes[1];
    int dimension = std::max(rows, cols);
    
    // Calculate the scaling factors for rows and columns to achieve a square aspect ratio
    float scale_row = static_cast<float>(dimension) / rows;
    float scale_col = static_cast<float>(dimension) / cols;
    
    // Print the grid with layout
    for (int y = 0; y < dimension; ++y) {
        if (y % int(scale_row) == 0) {
            for (int x = 0; x < dimension; ++x) {
                buffer << "+---";
            }
            buffer << "+" << std::endl;
        }
        
        for (int x = 0; x < dimension; ++x) {
            if (x % int(scale_col) == 0) {
                buffer << "|";
            }
            
            int orig_x = static_cast<int>(x / scale_col);
            int orig_y = static_cast<int>(y / scale_row);
            
            if (orig_x < cols && orig_y < rows) {
                float value = grid[orig_y][orig_x].item<float>();
                
                if (value == 1) {
                    buffer << " R ";
                } else if (value == 2) {
                    buffer << " C ";
                } else if (value == 3) {
                    buffer << " B ";
                } else {
                    buffer << "   ";
                }
            } else {
                buffer << "   "; // Ensure consistent grid cell size
            }
        }
        
        buffer << "|\n"; // Complete the grid row
    }
    
    // Print the bottom grid border
    for (int x = 0; x < dimension; ++x) {
        buffer << "+---";
    }
    buffer << "+\n";
    buffer << "Time: " << step * 0.1 << " seconds" << std::endl;
    
    // Print the entire grid at once
    std::cout << buffer.str();
    std::cout.flush(); // Ensure the output is rendered immediately
    
    usleep(100000); // Sleep for 0.1 seconds (100,000 microseconds)
}

void AACLoopFunction::SetTraining(bool value) {
    training = value;
}

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");

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
  m_fObjectiveFunction = 0;
  fTimeStep = 0;

  policy_param.resize(size_policy_net);
  std::fill(policy_param.begin(), policy_param.end(), 0.0f);
  value_param.resize(size_value_net);
  std::fill(value_param.begin(), value_param.end(), 0.0f);

  for (const auto& named_param : critic_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_critic[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
      }
  }

  for (const auto& named_param : actor_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_actor[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
      }
  }
    
  critic_net.to(device);
  actor_net.to(device);

  optimizer_critic->zero_grad();
  optimizer_actor->zero_grad();

  I = 1;

  TDerrors.resize(mission_lengh);
  std::fill(TDerrors.begin(), TDerrors.end(), 0.0f);

  Entropies.resize(mission_lengh);
  std::fill(Entropies.begin(), Entropies.end(), 0.0f);

  critic_losses.resize(mission_lengh);
  std::fill(critic_losses.begin(), critic_losses.end(), 0.0f);

  actor_losses.resize(mission_lengh);
  std::fill(actor_losses.begin(), actor_losses.end(), 0.0f);

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
  // critic_input_dim *= nb_robots;
  GetNodeAttribute(criticParameters, "hidden_dim", critic_hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", critic_num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", critic_output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);
  GetNodeAttribute(criticParameters, "alpha_critic", alpha_critic);
  GetNodeAttribute(criticParameters, "gamma", gamma);
  GetNodeAttribute(criticParameters, "device", device_type);
  device = (device_type == "cuda") ? torch::kCUDA : torch::kCPU;
  GetNodeAttribute(criticParameters, "port", port);

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
  GetNodeAttribute(experimentNode, "length", mission_lengh);
  mission_lengh *= 10;
  
  fTimeStep = 0;

  m_context = zmq::context_t(1);
  m_socket_actor = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_actor.connect("tcp://localhost:" + std::to_string(port));

  m_socket_critic = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_critic.connect("tcp://localhost:" + std::to_string(port+1));

  if(actor_type == "dandel"){
    // Dandel
    if(actor_num_hidden_layers>0){
      size_policy_net = (actor_input_dim*actor_hidden_dim+actor_hidden_dim) + (actor_num_hidden_layers-1)*(actor_hidden_dim*actor_hidden_dim+actor_hidden_dim) + 2*(actor_hidden_dim*actor_output_dim+actor_output_dim);
    }else{
      size_policy_net = actor_input_dim*actor_output_dim + 2 * actor_output_dim;
    }
  }else{
    // // Daisy
    // if(actor_num_hidden_layers>0){
    //   size_policy_net = (actor_input_dim*actor_hidden_dim+actor_hidden_dim) + (actor_num_hidden_layers-1)*(actor_hidden_dim*actor_hidden_dim+actor_hidden_dim) + (actor_hidden_dim*actor_output_dim+actor_output_dim) + 6 * (actor_hidden_dim + 1);
    // }else{
    //   size_policy_net = actor_input_dim*actor_output_dim + actor_output_dim + 6 * (actor_input_dim + 1);
    // }

    // Daisy
    if(actor_num_hidden_layers>0){
      size_policy_net = (actor_input_dim*actor_hidden_dim+actor_hidden_dim) + (actor_num_hidden_layers-1)*(actor_hidden_dim*actor_hidden_dim+actor_hidden_dim) + (actor_hidden_dim*actor_output_dim+actor_output_dim);
    }else{
      size_policy_net = actor_input_dim*actor_output_dim + actor_output_dim;
    }
  }

  if(critic_num_hidden_layers>0){
    size_value_net = (critic_input_dim*critic_hidden_dim+critic_hidden_dim) + (critic_num_hidden_layers-1)*(critic_hidden_dim*critic_hidden_dim+critic_hidden_dim) + (critic_hidden_dim*critic_output_dim+critic_output_dim);
  }else{
    size_value_net = critic_input_dim*critic_output_dim + critic_output_dim;
  }

  critic_net = Critic_Net(critic_input_dim, critic_hidden_dim, critic_num_hidden_layers, critic_output_dim);
  actor_net = argos::CEpuckNNController::Daisy(actor_input_dim, actor_hidden_dim, actor_num_hidden_layers, actor_output_dim);
  // actor_net = argos::CEpuckNNController::Dandel(actor_input_dim, actor_hidden_dim, actor_num_hidden_layers, actor_output_dim);
  
  critic_net.to(device);
  actor_net.to(device);

  optimizer_actor = std::make_shared<torch::optim::Adam>(actor_net.parameters(), torch::optim::AdamOptions(alpha_critic));
  optimizer_actor->zero_grad();
  optimizer_critic = std::make_shared<torch::optim::Adam>(critic_net.parameters(), torch::optim::AdamOptions(alpha_actor));
  optimizer_critic->zero_grad();

  policy_param.resize(size_policy_net);
  std::fill(policy_param.begin(), policy_param.end(), 0.0f);
  value_param.resize(size_value_net);
  std::fill(value_param.begin(), value_param.end(), 0.0f);

  // std::cout << "size_policy_net = " <<  size_policy_net << std::endl;
  // std::cout << "size_value_net = " <<  size_value_net << std::endl;


  for (const auto& named_param : critic_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_critic[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
      }
  }

  for (const auto& named_param : actor_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_actor[named_param.key()] = torch::zeros_like(named_param.value()).to(device);
      }
  }

  TDerrors.resize(mission_lengh);
  std::fill(TDerrors.begin(), TDerrors.end(), 0.0f);

  Entropies.resize(mission_lengh);
  std::fill(Entropies.begin(), Entropies.end(), 0.0f);

  critic_losses.resize(mission_lengh);
  std::fill(critic_losses.begin(), critic_losses.end(), 0.0f);

  actor_losses.resize(mission_lengh);
  std::fill(actor_losses.begin(), actor_losses.end(), 0.0f);

  I = 1;
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
    
    // std::cout << "state = " <<  state << std::endl;

    // Load up-to-date value network
    std::vector<float> critic;
    m_value->mutex.lock();
    for (auto w : m_value->vec) {
        critic.push_back(w);
    }
    m_value->mutex.unlock();

    // Update all parameters with values from the new_params vector
    size_t index = 0;
    for (auto& param : critic_net.parameters()) {
      auto param_data = torch::from_blob(critic.data() + index, param.data().sizes()).clone();
      param.data() = param_data;
      index += param_data.numel();
    }

    // Load up-to-date policy network
    std::vector<float> actor;
    m_policy->mutex.lock();
    for(auto w : m_policy->vec){
      actor.push_back(w);
    }
    m_policy->mutex.unlock();

    // Update all parameters with values from the new_params vector
    index = 0;
    for (auto& param : actor_net.parameters()) {
      auto param_data = torch::from_blob(actor.data() + index, param.data().sizes()).clone();
      param.data() = param_data;
      index += param_data.numel();
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
        critic_net.to(device);
        torch::Tensor v_state = critic_net.forward(state_batch);
        torch::Tensor v_state_prime = critic_net.forward(next_state_batch);
        

        // Adjust for the terminal state
        if (fTimeStep + 1 == mission_lengh) {
            v_state_prime = torch::zeros_like(v_state_prime);
        }

        // Global reward
        // LOG << "rewards = " <<  rewards << std::endl;

        // TD error
        torch::Tensor reward_batch = torch::stack(rewards).unsqueeze(1).to(device);
        torch::Tensor td_errors = reward_batch + gamma * v_state_prime - v_state;

        // std::cout << "state_batch = " <<  state_batch << std::endl;
        // std::cout << "next_state_batch = " <<  next_state_batch << std::endl;
        // std::cout << "v_state = " <<  v_state << std::endl;
        // std::cout << "v_state_prime = " <<  v_state_prime << std::endl;
        // std::cout << "reward_batch = " <<  reward_batch << std::endl;
        // std::cout << "td_errors = " <<  td_errors << std::endl;

        
        TDerrors[fTimeStep] = td_errors.mean().item<float>();

        // Critic Loss
        torch::Tensor critic_loss = td_errors.pow(2).mean();
        // std::cout << "critic_loss = " <<  critic_loss << std::endl;
        critic_losses[fTimeStep] = critic_loss.item<float>();

        // Backward and optimize for critic
        optimizer_critic->zero_grad();
        critic_loss.backward();
        // Update eligibility traces and gradients
        for (auto& named_param : critic_net.named_parameters()) {
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
        }
        // std::cout << "log_probs = " <<  log_probs << std::endl;
        // std::cout << "entropies = " <<  entropies << std::endl;

        // Calculate policy loss using the per-robot TD error times the logprob
        torch::Tensor log_probs_batch = torch::stack(log_probs).unsqueeze(1);
        torch::Tensor policy_loss = -(log_probs_batch * td_errors.detach()).mean();

        // Add entropy term for better exploration
        torch::Tensor entropy_batch = torch::stack(entropies).unsqueeze(1);
        policy_loss -= entropy_fact * entropy_batch.mean();
        // std::cout << "log_probs_batch = " <<  log_probs_batch << std::endl;
        // std::cout << "entropy_batch = " <<  entropy_batch << std::endl;
        // std::cout << "policy_loss = " <<  policy_loss << std::endl;
        actor_losses[fTimeStep] = policy_loss.item<float>();
        Entropies[fTimeStep] = entropy_batch.mean().item<float>();

        // Backward and optimize for actor
        actor_net.to(device);
        optimizer_actor->zero_grad();
        policy_loss.backward();
        // Update eligibility traces and gradients
        for (auto& named_param : actor_net.named_parameters()) {
            auto& param = named_param.value();
            if (param.grad().defined()) {
                auto& eligibility = eligibility_trace_actor[named_param.key()];
                eligibility.mul_(gamma * lambda_actor).add_(param.grad() * I);
                param.mutable_grad() = eligibility.clone();
            }
            // std::cout << "actor " <<  named_param.key() << " grad: " << param.grad() << std::endl;
        }
        optimizer_actor->step();

        I *= gamma; // Update the decay term for the eligibility traces
      }catch (std::exception &ex) {
        LOGERR << "Error in training loop: " << ex.what() << std::endl;
      }
      GetParametersVector(actor_net, policy_param);
      GetParametersVector(critic_net, value_param);

      // actor
      // std::cout << "policy_param: " << policy_param << std::endl;
      Data policy_data {policy_param};   
      std::string serialized_data = serialize(policy_data);
      zmq::message_t message(serialized_data.size());
      memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
      m_socket_actor.send(message, zmq::send_flags::dontwait);
      // critic
      // std::cout << "value_param: " << value_param << std::endl;
      Data value_data {value_param};   
      serialized_data = serialize(value_data);
      message.rebuild(serialized_data.size());
      memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
      m_socket_critic.send(message, zmq::send_flags::dontwait);  
    }
    
    states.clear();
    for (const auto& state : positions)
    {
      states.push_back(state.clone());
    }

    // Load up-to-date value network
    std::vector<float> critic;
    m_value->mutex.lock();
    for (auto w : m_value->vec) {
        critic.push_back(w);
    }
    m_value->mutex.unlock();

    // Update all parameters with values from the new_params vector
    size_t index = 0;
    for (auto& param : critic_net.parameters()) {
      auto param_data = torch::from_blob(critic.data() + index, param.data().sizes()).clone();
      param.data() = param_data;
      index += param_data.numel();
    }

    // Load up-to-date policy network
    std::vector<float> actor;
    m_policy->mutex.lock();
    for(auto w : m_policy->vec){
      actor.push_back(w);
    }
    m_policy->mutex.unlock();

    // Update all parameters with values from the new_params vector
    index = 0;
    for (auto& param : actor_net.parameters()) {
      auto param_data = torch::from_blob(actor.data() + index, param.data().sizes()).clone();
      param.data() = param_data;
      index += param_data.numel();
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

void AACLoopFunction::GetParametersVector(const torch::nn::Module& module, std::vector<float>& params_vector) {
    // Calculate total number of parameters
    int64_t total_params = 0;
    for (const auto& p : module.parameters()) {
        total_params += p.numel();
    }

    // Clear the vector and reserve space to avoid reallocations
    params_vector.clear();
    params_vector.reserve(total_params);

    // Iterate through all parameters and add their elements to the vector
    for (const auto& p : module.parameters()) {
        // Ensure the tensor is on the CPU before accessing its data
        auto p_cpu = p.data().view(-1).to(torch::kCPU);
        auto p_data_ptr = p_cpu.data_ptr<float>();
        auto num_elements = p_cpu.numel();
        for (int64_t i = 0; i < num_elements; ++i) {
            params_vector.push_back(p_data_ptr[i]);
        }
    }
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

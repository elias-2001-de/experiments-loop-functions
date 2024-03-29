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
  m_fRadius = 0.6;
  m_cCoordBlackSpot = CVector2(0,-0.6);
  //m_cCoordWhiteSpot = CVector2(0,0.6);
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
          eligibility_trace_critic[named_param.key()] = torch::zeros_like(named_param.value());
      }
  }

  for (const auto& named_param : actor_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_actor[named_param.key()] = torch::zeros_like(named_param.value());
      }
  }
    

  optimizer_critic->zero_grad();
  optimizer_actor->zero_grad();

  I = 1;

  CoreLoopFunctions::Reset();
}

/****************************************/
/****************************************/

void AACLoopFunction::Init(TConfigurationNode& t_tree) {
  CoreLoopFunctions::Init(t_tree);
  TConfigurationNode cParametersNode;
  try {
     cParametersNode = GetNode(t_tree, "params");
  } catch(std::exception e) {
  }
  GetNodeAttribute(cParametersNode, "number_robots", nb_robots);

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", critic_input_dim);
  GetNodeAttribute(criticParameters, "hidden_dim", critic_hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", critic_num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", critic_output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);
  GetNodeAttribute(criticParameters, "alpha_critic", alpha_critic);
  GetNodeAttribute(criticParameters, "gamma", gamma);
  GetNodeAttribute(criticParameters, "port", port);

  TConfigurationNode actorParameters;
  argos::TConfigurationNode& parentNode = *dynamic_cast<argos::TConfigurationNode*>(t_tree.Parent());
  TConfigurationNode controllerNode;
  TConfigurationNode dandelNode;
  TConfigurationNode actorNode;
  controllerNode = GetNode(parentNode, "controllers");
  dandelNode = GetNode(controllerNode, "dandel_controller");
  actorParameters = GetNode(dandelNode, "actor");
  GetNodeAttribute(actorParameters, "input_dim", actor_input_dim);
  GetNodeAttribute(actorParameters, "hidden_dim", actor_hidden_dim);
  GetNodeAttribute(actorParameters, "num_hidden_layers", actor_num_hidden_layers);
  GetNodeAttribute(actorParameters, "output_dim", actor_output_dim);
  GetNodeAttribute(actorParameters, "lambda_actor", lambda_actor);
  GetNodeAttribute(actorParameters, "alpha_actor", alpha_actor);

  TConfigurationNode frameworkNode;
  TConfigurationNode experimentNode;
  frameworkNode = GetNode(parentNode, "framework");
  experimentNode = GetNode(frameworkNode, "experiment");
  GetNodeAttribute(experimentNode, "length", mission_lengh);
  
  fTimeStep = 0;

  m_context = zmq::context_t(1);
  m_socket_actor = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_actor.connect("tcp://localhost:" + std::to_string(port));

  m_socket_critic = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_critic.connect("tcp://localhost:" + std::to_string(port+1));

  if(actor_num_hidden_layers>0){
    size_policy_net = (actor_input_dim*actor_hidden_dim+actor_hidden_dim) + (actor_num_hidden_layers-1)*(actor_hidden_dim*actor_hidden_dim+actor_hidden_dim) + (actor_hidden_dim*actor_output_dim+actor_output_dim);
  }else{
    size_policy_net = actor_input_dim*actor_output_dim + actor_output_dim;
  }

  if(critic_num_hidden_layers>0){
    size_value_net = (critic_input_dim*critic_hidden_dim+critic_hidden_dim) + (critic_num_hidden_layers-1)*(critic_hidden_dim*critic_hidden_dim+critic_hidden_dim) + (critic_hidden_dim*critic_output_dim+critic_output_dim);
  }else{
    size_value_net = critic_input_dim*critic_output_dim + critic_output_dim;
  }

  critic_net = Critic_Net(critic_input_dim, critic_hidden_dim, critic_num_hidden_layers, critic_output_dim);
  actor_net = argos::CEpuckNNController::Actor_Net(actor_input_dim, actor_hidden_dim, actor_num_hidden_layers, actor_output_dim);

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

  for (const auto& named_param : critic_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_critic[named_param.key()] = torch::zeros_like(named_param.value());
      }
  }

  for (const auto& named_param : actor_net.named_parameters()) {
      if (named_param.value().requires_grad()) {
          eligibility_trace_actor[named_param.key()] = torch::zeros_like(named_param.value());
      }
  }

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

  // d = (m_cCoordWhiteSpot - vCurrentPoint).Length();
  // if (d <= m_fRadius) {
  //   return CColor::WHITE;
  // }

  return CColor::GRAY50;
}

/****************************************/
/****************************************/

void AACLoopFunction::PreStep() {
  // std::cout << "Prestep\n";
  // Observe state S
  /* Check position for each agent. 
   * If robot is in cell i --> 1
   * else 0
   */
  at::Tensor grid = torch::zeros({50, 50});
  torch::Tensor pos = torch::empty({critic_input_dim});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  CRadians cRoll, cPitch, cYaw;
  int pos_index = 0;
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

    cEpuckOrientation = pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);
    
    float cEpuckAngle = M_PI/2 + cYaw.GetValue();
    float angleCos = std::cos(cEpuckAngle);
    float angleSin = std::sin(cEpuckAngle); 

    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));

    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;

    // PART1: State is the position of the single agent
    pos = pos.index_put_({pos_index}, cEpuckPosition.GetX());
    pos = pos.index_put_({pos_index+1}, cEpuckPosition.GetY());
    pos = pos.index_put_({pos_index+2}, std::cos(cYaw.GetValue()));
    pos = pos.index_put_({pos_index+3}, std::sin(cYaw.GetValue()));
    pos_index += 4;
  }

  state = pos.clone();
  // std::cout << "state: " << state << std::endl;
  // Flatten the 2D tensor to 1D for net input
  //state = grid.view({1, 1, 50, 50}).clone();

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
  for (CSpace::TMapPerType::iterator it = cEntities.begin();
       it != cEntities.end(); ++it) {
    CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
    try {
      CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
      cController.SetNetworkAndOptimizer(actor_net, optimizer_actor);
    } catch (std::exception &ex) {
      LOGERR << "Error while setting network: " << ex.what() << std::endl;
    }
  }
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  // std::cout << "Post step\n";
  at::Tensor grid = torch::zeros({50, 50});
  torch::Tensor pos = torch::empty({critic_input_dim});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  CRadians cRoll, cPitch, cYaw;
  std::vector<int> rewards(nb_robots, 0);
  int pos_index = 0;
  int i = 0;
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                       
    cEpuckOrientation = pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);
    
    float cEpuckAngle = M_PI/2 + cYaw.GetValue();
    float angleCos = std::cos(cEpuckAngle);
    float angleSin = std::sin(cEpuckAngle); 

    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));

    Real fDistanceSpot = (m_cCoordBlackSpot - cEpuckPosition).Length();
    if (fDistanceSpot <= m_fRadius) {
      m_fObjectiveFunction += 1;
      rewards[i] = 1;
    }else{
      rewards[i] = 0;
    }

    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;

    // pos = pos.index_put_({pos_index}, grid_x);
    // pos = pos.index_put_({pos_index+1}, grid_y);
    // pos = pos.index_put_({pos_index+2}, angleCos);
    // pos = pos.index_put_({pos_index+3}, angleSin);
    pos = pos.index_put_({pos_index}, cEpuckPosition.GetX());
    pos = pos.index_put_({pos_index+1}, cEpuckPosition.GetY());
    pos = pos.index_put_({pos_index+2}, std::cos(cYaw.GetValue()));
    pos = pos.index_put_({pos_index+3}, std::sin(cYaw.GetValue()));
    pos_index += 4;
    i++;
  }
  state_prime = pos.clone();

  // std::cout << "score: " << m_fObjectiveFunction << std::endl;	  
  // place spot in grid
  int grid_x = static_cast<int>(std::round((m_cCoordBlackSpot.GetX() + 1.231) / 2.462 * 49));
  int grid_y = static_cast<int>(std::round((-m_cCoordBlackSpot.GetY() + 1.231) / 2.462 * 49));
  int radius = static_cast<int>(std::round(m_fRadius / 2.462 * 49));
  // Print arena on the terminal
  auto temp1 = grid[grid_x][grid_y];
  auto temp2 = grid[grid_x+radius][grid_y];
  auto temp3 = grid[grid_x-radius][grid_y];
  auto temp4 = grid[grid_x][grid_y+radius];
  auto temp5 = grid[grid_x][grid_y-radius];
  grid[grid_x][grid_y] = 2;
  grid[grid_x+radius][grid_y] = 3;
  grid[grid_x-radius][grid_y] = 3;
  grid[grid_x][grid_y+radius] = 3;
  grid[grid_x][grid_y-radius] = 3;
  // std::cout << "GRID State:\n";
  // print_grid(grid);
  grid[grid_x][grid_y] = temp1;
  grid[grid_x+radius][grid_y] = temp2;
  grid[grid_x-radius][grid_y] = temp3;
  grid[grid_x][grid_y+radius] = temp4;
  grid[grid_x][grid_y-radius] = temp5;
  //}
  //m_fObjectiveFunction = m_unScoreSpot1/(Real) m_unNumberRobots;
  //LOG << "Score = " << m_fObjectiveFunction << std::endl;

  // Observe state S'
  /* Check position for each agent. 
   * If robot is in grid --> 1
   * else 0
   */

  // Compute delta with Critic predictions
  if (training) {
    // Assuming 'state', 'state_prime', 'rewards', 'device', 'gamma', 'lambda_critic', 'lambda_actor', and 'I'
    // are already defined and initialized appropriately.

    state = state.to(device);
    state_prime = state_prime.to(device);

    // Forward passes
    torch::Tensor v_state = critic_net.forward(state);
    torch::Tensor v_state_prime = critic_net.forward(state_prime);

    // Adjust for the terminal state
    if (fTimeStep + 1 == mission_lengh * 10) {
        v_state_prime = torch::zeros_like(v_state_prime);
    }

    // Global reward
    float total_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0f);
    torch::Tensor total_reward_tensor = torch::tensor({total_reward}, torch::dtype(torch::kFloat32).device(device));

    // TD error
    torch::Tensor delta = total_reward_tensor + gamma * v_state_prime - v_state;

    // Critic Loss
    torch::Tensor critic_loss = delta.pow(2).mean();

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

    // Compute individual TD errors for policy loss calculation
    torch::Tensor rewards_tensor = torch::tensor(rewards, torch::dtype(torch::kFloat32).device(device)).unsqueeze(1); // [nb_robots, 1]

    torch::Tensor individual_delta = rewards_tensor + gamma * v_state_prime/nb_robots - v_state/nb_robots; // [nb_robots, 1]

    // Prepare for policy update
    std::vector<torch::Tensor> log_probs;
    CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller"); 
    for (CSpace::TMapPerType::iterator it = cEntities.begin();
        it != cEntities.end(); ++it) {
      CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);      
      CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
      log_probs.push_back(cController.GetPolicyLogProbs()); 
    }

    // Calculate policy loss using the per-robot TD error times the logprob
    torch::Tensor policy_loss = -(torch::stack(log_probs) * individual_delta.detach()).sum(); // Sum of element-wise product

    // Backward and optimize for actor
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
    }
    optimizer_actor->step();

    I *= gamma; // Update the decay term for the eligibility traces

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
    
    fTimeStep += 1;

    // std::cout << "value_param " << value_param << std::endl;
    // std::cout << "value_param size " << value_param.size() << std::endl;
    // std::cout << "policy_param " << policy_param << std::endl;
    // std::cout << "policy_param size " << policy_param.size() << std::endl;
  }
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

CVector3 AACLoopFunction::GetRandomPosition() {

  // Real temp;
  // Real a = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));
  // Real  b = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));

  // Real fPosX = a * 0.3 + 0.3;
  // Real fPosY = b * 0.6;

  // //std::cout << "CVector3(" << fPosX << "," << fPosY << ", 0)," << std::endl;
  // return CVector3(fPosX, fPosY, 0);
  
  // Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  // Real  b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  
  // Real fPosX = std::max(b * m_fDistributionRadius * cos(2 * CRadians::PI.GetValue() * (a/b)), 0.0);
  // Real fPosY = b * m_fDistributionRadius * sin(2 * CRadians::PI.GetValue() * (a/b));

  // Generate angle in range [Pi, 2*Pi] for the left half of the disk
  Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  Real b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));

  // Generate angle in the range [Pi, 2*Pi] for the left half of the disk
  Real angle = CRadians::PI.GetValue() * 0.75f + (CRadians::PI.GetValue() * 0.5f * a);

  Real radius = b * m_fDistributionRadius;

  Real fPosX = -radius * cos(angle);  // This will be negative or zero, for the left half of the disk
  Real fPosY = radius * sin(angle);  // Y position

  return CVector3(fPosY, fPosX, 0);

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
        auto p_data_ptr = p.data().view(-1).data_ptr<float>();
        auto num_elements = p.numel();
        for (int64_t i = 0; i < num_elements; ++i) {
            params_vector.push_back(p_data_ptr[i]);
        }
    }
}


/****************************************/
/****************************************/

void AACLoopFunction::print_grid(at::Tensor grid){
    // Get the size of the grid tensor
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
            // Print the top and bottom grid borders
            for (int x = 0; x < dimension; ++x) {
                std::cout << "+---";
            }
            std::cout << "+" << std::endl;
        }

        for (int x = 0; x < dimension; ++x) {
            if (x % int(scale_col) == 0) {
                std::cout << "|"; // Print the left grid border
            }

            // Calculate the corresponding position in the original grid
            int orig_x = static_cast<int>(x / scale_col);
            int orig_y = static_cast<int>(y / scale_row);

            // Check if the current position is within the original grid bounds
            if (orig_x < cols && orig_y < rows) {
                float value = grid[orig_y][orig_x].item<float>();

                // Print highlighted character if the value is 1, otherwise print a space
                if (value == 1) {
                    std::cout << " R ";
		} else if (value == 2){
		    std::cout << " C ";
		} else if (value == 3){
                    std::cout << " B ";
                } else {
                    std::cout << "   ";
                }
            } else {
                std::cout << " "; // Print a space for out-of-bounds positions
            }
        }

        if (y % int(scale_row) == 0) {
            std::cout << "|" << std::endl; // Print the right grid border
        } else {
            std::cout << "|" << std::endl; // Print a separator for other rows
        }
    }

    // Print the bottom grid border
    for (int x = 0; x < dimension; ++x) {
        std::cout << "+---";
    }
    std::cout << "+" << std::endl;
}

void AACLoopFunction::SetTraining(bool value) {
    training = value;
}

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");

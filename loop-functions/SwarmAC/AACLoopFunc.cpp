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

  optimizer_critic->zero_grad();
  optimizer_actor->zero_grad();

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

  optimizer_actor = std::make_shared<torch::optim::Adam>(actor_net.parameters(), torch::optim::AdamOptions(lambda_actor));
  optimizer_actor->zero_grad();
  optimizer_critic = std::make_shared<torch::optim::Adam>(critic_net.parameters(), torch::optim::AdamOptions(lambda_critic));
  optimizer_critic->zero_grad();

  policy_param.resize(size_policy_net);
  std::fill(policy_param.begin(), policy_param.end(), 0.0f);
  value_param.resize(size_value_net);
  std::fill(value_param.begin(), value_param.end(), 0.0f);
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
  if(training){

    state.set_requires_grad(true);
    state = state.to(device);
    state_prime = state_prime.to(device);
    torch::Tensor v_state = critic_net.forward(state);
    torch::Tensor v_state_prime = critic_net.forward(state_prime);
    if(fTimeStep+1 == mission_lengh*10) v_state_prime = torch::tensor({0}).view({1, 1});
    float reward = 0;
    for (auto& n : rewards) reward += n;
    delta = reward + (gamma * v_state_prime[0]) - v_state[0];
    auto critic_loss = delta*delta;
    std::vector<torch::Tensor> td_errors(nb_robots);
    for (int i = 0; i < rewards.size(); i++)
      td_errors[i] = rewards[i] + (gamma * v_state_prime[0]) - v_state[0];
    
    if(fTimeStep+1 <= mission_lengh*10){
      // std::cout << "TimeStep (prime) = " << fTimeStep+1 << std::endl;
      // std::cout << "state = " << state << std::endl;
      // std::cout << "v(s) = " << v_state[0].item<float>() << std::endl;
      // std::cout << "v(s') = " << v_state_prime[0].item<float>() << std::endl;
      // std::cout << "gamma * v(s') = " << gamma * v_state_prime[0].item<float>() << std::endl;
      // std::cout << "reward = " << reward << std::endl;
      // std::cout << "delta = " << delta << std::endl;
      // std::cout << "score = " << m_fObjectiveFunction << std::endl;
    }
  
    optimizer_critic->zero_grad();
    critic_loss.backward();
    auto critic_params = critic_net.parameters();
    for (size_t i = 0; i < critic_params.size(); ++i) {
        auto& param = critic_params[i];
        // Ensure there is a gradient to work with
        if (param.grad().defined()) {
            if (i < value_trace.size()) {
                value_trace[i].mul_(gamma * lambda_critic).add_(param.grad().data());
            } else {
                // Initialize value_trace[i] if it doesn't already exist
                value_trace.push_back(param.grad().clone().mul(gamma * lambda_critic));
            }
            param.mutable_grad() = value_trace[i].clone();
        }
    }

    // Step the optimizer to update the model parameters
    optimizer_critic->step();

    int rb_iter = 0;
    std::vector<torch::Tensor> log_probs(nb_robots);
    CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller"); 
    for (CSpace::TMapPerType::iterator it = cEntities.begin();
        it != cEntities.end(); ++it) {
      CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);	    
      CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
      log_probs[rb_iter] = cController.GetPolicyLogProbs();	
      rb_iter++;
    }

    // Ensure log_probs and td_errors have the same size.
    if (log_probs.size() != td_errors.size()) {
        throw std::runtime_error("log_probs and td_errors vectors must be the same size.");
    }

    // Calculate policy loss
    // Initialize a tensor to accumulate policy loss
    torch::Tensor policy_loss = torch::tensor(0.0, torch::requires_grad(true));
    for (size_t i = 0; i < log_probs.size(); ++i) {
        // Assuming log_probs and td_errors are directly related and of the same size
        torch::Tensor contribution = log_probs[i] * td_errors[i].detach();
        policy_loss = policy_loss + contribution;
    }

    // Negate the final result.
    policy_loss = -policy_loss;

    optimizer_actor->zero_grad();
    policy_loss.backward();
    auto actor_params = actor_net.parameters();
    for (size_t i = 0; i < actor_params.size(); ++i) {
        auto& param = actor_params[i];
        // Ensure there is a gradient to work with
        if (param.grad().defined()) {
            if (i < policy_trace.size()) {
                policy_trace[i].mul_(gamma * lambda_actor).add_(param.grad().data());
            } else {
                // Initialize value_trace[i] if it doesn't already exist
                policy_trace.push_back(param.grad().clone().mul(gamma * lambda_actor));
            }
            param.mutable_grad() = policy_trace[i].clone();
        }
    }
    optimizer_actor->step();

    // Lambda to extract and flatten parameters from a given module
    auto extract_and_flatten_critic_parameters = [&](const torch::nn::Module& module) {
      for (const auto& p : module.parameters()) {
          if (p.defined()) {
              // First, flatten the tensor and store it in a variable to extend its lifetime
              auto p_flat = p.view(-1);

              // Now, obtain an accessor on the non-temporary tensor
              auto p_accessor = p_flat.accessor<float, 1>();

              // You can now safely use p_accessor to access the tensor's data
              for(int64_t i = 0; i < p_accessor.size(0); ++i) {
                  value_param.push_back(p_accessor[i]);
              }
          }
      }
    };

    // Extract and flatten parameters from the network
    extract_and_flatten_critic_parameters(critic_net);


    // Lambda to extract and flatten parameters from a given module
    auto extract_and_flatten_actor_parameters = [&](const torch::nn::Module& module) {
      for (const auto& p : module.parameters()) {
          if (p.defined()) {
              // First, flatten the tensor and store it in a variable to extend its lifetime
              auto p_flat = p.view(-1);

              // Now, obtain an accessor on the non-temporary tensor
              auto p_accessor = p_flat.accessor<float, 1>();

              // You can now safely use p_accessor to access the tensor's data
              for(int64_t i = 0; i < p_accessor.size(0); ++i) {
                  policy_param.push_back(p_accessor[i]);
              }
          }
      }
    };

    // Extract and flatten parameters from the network
    extract_and_flatten_actor_parameters(actor_net);

    // actor
    Data policy_data {policy_param};   
    std::string serialized_data = serialize(policy_data);
    zmq::message_t message(serialized_data.size());
    memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
    m_socket_actor.send(message, zmq::send_flags::dontwait);
    // critic
    Data value_data {value_param};   
    serialized_data = serialize(value_data);
    message.rebuild(serialized_data.size());
    memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
    m_socket_critic.send(message, zmq::send_flags::dontwait);  
    fTimeStep += 1;
  }
  std::cout << "step: " << fTimeStep << std::endl;
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

// Assuming actor_net is your Actor Neural Network (an instance of torch::nn::Module)
// Assuming behavior_probs, beta_parameters, chosen_behavior, chosen_parameter, and td_error are already computed

void AACLoopFunction::update_actor(torch::nn::Module& actor_net, torch::Tensor& behavior_probs, 
                  torch::Tensor& beta_parameters, int chosen_behavior, 
                  double chosen_parameter, double td_error, 
                  torch::optim::Optimizer& optimizer) {

    // Compute log-probability of chosen behavior
    auto log_prob_behavior = torch::log(behavior_probs[chosen_behavior]);

    // Compute parameters for Beta distribution
    auto alpha = beta_parameters[chosen_behavior][0].item<double>();
    auto beta = beta_parameters[chosen_behavior][1].item<double>();

    // Compute log-PDF of Beta distribution at the chosen parameter
    auto log_pdf_beta = computeBetaLogPDF(alpha, beta, chosen_parameter);

    // Compute the surrogate objective
    auto loss = - (log_prob_behavior + log_pdf_beta) * td_error;  // The minus sign because we want to do gradient ascent

    // Zero gradients
    optimizer.zero_grad();

    // Compute gradients w.r.t. actor network parameters (weight and biases)
    loss.backward();

    // Perform one step of the optimization (gradient descent)
    optimizer.step();
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

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

  delta = 0;

  policy_trace.resize(size_policy_net);
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
  value_trace.resize(size_value_net);
  std::fill(value_trace.begin(), value_trace.end(), 0.0f);

  policy_update.resize(size_policy_net);
  std::fill(policy_update.begin(), policy_update.end(), 0.0f);
  value_update.resize(size_value_net);
  std::fill(value_update.begin(), value_update.end(), 0.0f);

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

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", input_dim);
  GetNodeAttribute(criticParameters, "hidden_dim", hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);
  GetNodeAttribute(criticParameters, "gamma", gamma);
  GetNodeAttribute(criticParameters, "port", port);
    

  // std::cout << "[ARGoS] Critic input_dim: " << input_dim << std::endl;
  // std::cout << "[ARGoS] Critic hidden_dim: " << hidden_dim << std::endl;
  // std::cout << "[ARGoS] Critic num_hidden_layers: " << num_hidden_layers << std::endl;
  // std::cout << "[ARGoS] Critic output_dim: " << output_dim << std::endl;
  // std::cout << "[ARGoS] Critic port: " << port << std::endl;
  
  fTimeStep = 0;

  m_context = zmq::context_t(1);
  m_socket_actor = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_actor.connect("tcp://localhost:" + std::to_string(port));

  m_socket_critic = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_critic.connect("tcp://localhost:" + std::to_string(port+1));

  // Initialise traces, delta and adam update
  delta = 0;

  size_value_net = (input_dim*hidden_dim+hidden_dim) + (num_hidden_layers-1)*(hidden_dim*hidden_dim+hidden_dim) + (hidden_dim*output_dim+output_dim);
  size_policy_net = (input_dim*hidden_dim+hidden_dim) + (num_hidden_layers-1)*(hidden_dim*hidden_dim+hidden_dim) + (4*hidden_dim+4); // to be changed for more flexibility
  // std::cout << "[ARGoS] size_value_net: " << size_value_net << std::endl;
  // std::cout << "[ARGoS] size_policy_net: " << size_policy_net << std::endl;
  critic_net = Net(input_dim, hidden_dim, num_hidden_layers, output_dim);

  policy_trace.resize(size_policy_net);
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
  value_trace.resize(size_value_net);
  std::fill(value_trace.begin(), value_trace.end(), 0.0f);
  
  policy_update.resize(size_policy_net);
  std::fill(policy_update.begin(), policy_update.end(), 0.0f);
  value_update.resize(size_value_net);
  std::fill(value_update.begin(), value_update.end(), 0.0f);

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
  torch::Tensor pos = torch::empty({4});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  CRadians cRoll, cPitch, cYaw;
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

    cEpuckOrientation = pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);
    
    float cEpuckAngle = cYaw.GetValue();
    float angleCos = std::cos(cEpuckAngle);
    float angleSin = std::sin(cEpuckAngle); 

    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));

    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;

    // PART1: State is the position of the single agent
    pos = pos.index_put_({0}, grid_x);
    pos = pos.index_put_({1}, grid_y);
    pos = pos.index_put_({2}, angleCos);
    pos = pos.index_put_({3}, angleSin);
  }

  state = pos.clone();
  std::cout << "state: " << state << std::endl;
  // Flatten the 2D tensor to 1D for net input
  //state = grid.view({1, 1, 50, 50}).clone();

  // Load up-to-date value network
  std::vector<float> critic;
  std::vector<float> fc_input_weights;

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

  std::vector<float> actor;
  m_policy->mutex.lock();
  //std::cout << "size global policy: " << m_policy->vec.size() << std::endl;
  for(auto w : m_policy->vec){
    actor.push_back(w);
  }
  m_policy->mutex.unlock();
  // std::cout << "accumulate of policy weights: " << accumulate(actor.begin(),actor.end(),0.0) << std::endl;
  // std::cout << "global policy size: " << actor.size() << std::endl;
   
  // Launch the experiment with the correct random seed and network,
  // and evaluate the average fitness
  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
  for (CSpace::TMapPerType::iterator it = cEntities.begin();
       it != cEntities.end(); ++it) {
    CControllableEntity *pcEntity =
        any_cast<CControllableEntity *>(it->second);
    try {
      CEpuckNNController& cController =
      dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
      //std::cout << "LOADNET\n";
      cController.LoadNetwork(actor, state);
    } catch (std::exception &ex) {
      LOGERR << "Error while setting network: " << ex.what() << std::endl;
    }
  }
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  // std::cout << "Poststep\n";
  at::Tensor grid = torch::zeros({50, 50});
  torch::Tensor pos = torch::empty({4});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  int reward = 0;
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
                       
    CRadians cRoll, cPitch, cYaw;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);
    float cEpuckAngle = cYaw.GetValue();
    float angleCos = std::cos(cEpuckAngle);
    float angleSin = std::sin(cEpuckAngle); 

    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));

    Real fDistanceSpot = (m_cCoordBlackSpot - cEpuckPosition).Length();
    if (fDistanceSpot <= m_fRadius) {
      m_fObjectiveFunction += 1;
      reward += 1;
    }

    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;

    pos = pos.index_put_({0}, grid_x);
    pos = pos.index_put_({1}, grid_y);
    pos = pos.index_put_({2}, angleCos);
    pos = pos.index_put_({3}, angleSin);
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
  auto device_type = torch::kCPU;
  critic_net.to(device_type);

  // Flatten the 2D tensor to 1D for net input
  //state_prime = grid.view({1, 1, 50, 50}).clone().to(device_type);

  // Compute delta with Critic predictions
  state = state.to(device_type);
  state.set_requires_grad(true);
  state_prime = state_prime.to(device_type);
  torch::Tensor v_state = critic_net.forward(state).to(device_type);
  torch::Tensor v_state_prime = critic_net.forward(state_prime).to(device_type);
  if(fTimeStep+1 == 1200)
	  v_state_prime = torch::tensor({0}).view({1, 1});
  delta = reward + (gamma * v_state_prime[0].item<float>()) - v_state[0].item<float>();
  
  if(fTimeStep >= 0){
    // std::cout << "TimeStep (prime) = " << fTimeStep+1 << std::endl;
    // std::cout << "state = " << state << std::endl;
    // std::cout << "v(s) = " << v_state[0].item<float>() << std::endl;
    // std::cout << "v(s') = " << v_state_prime[0].item<float>() << std::endl;
    // std::cout << "gamma * v(s') = " << gamma * v_state_prime[0].item<float>() << std::endl;
    // std::cout << "reward = " << reward << std::endl;
    // std::cout << "delta = " << delta << std::endl;
    // std::cout << "score = " << m_fObjectiveFunction << std::endl;
  }
 
  // Compute value trace
  v_state.to(device_type);
  v_state.requires_grad_(true);
  // Zero out the gradients
  critic_net.zero_grad();
  // Compute the gradient by back-propagation
  //critic_net.print_last_layer_params();
  //std::cout << "v_state = " << v_state << std::endl;
  v_state.backward();
  // Update value trace using gradients of each parameter
  int param_index = 0;
  for (const auto& parameter : critic_net.parameters()) {
    // Get the gradient tensor of the current parameter
    torch::Tensor gradients = parameter.grad().to(device_type);
    torch::Tensor flat_tensor = gradients.flatten().to(torch::kFloat32);
    std::vector<float> gradient_values(flat_tensor.numel());
    std::memcpy(gradient_values.data(), flat_tensor.data_ptr(), flat_tensor.numel() * sizeof(float));

    // Get the number of elements in the gradient tensor
    int num_elements = gradient_values.size();
    // Loop through each element of the gradient tensor
    for (int i = 0; i < num_elements; i++) {
        // Access the individual element of the gradient tensor
        //float gradient_value = gradients.view(-1)[i].item<float>();
        value_trace[param_index] = gamma * lambda_critic * value_trace[param_index] + gradient_values[i];

        // Move to the next index in value_trace
        param_index++;
    }

	  // Clear gradients for the current parameter to avoid interference with the next backward pass
	  parameter.grad().zero_();
  }

  // std::cout << "accumulate of value trace: " << accumulate(value_trace.begin(),value_trace.end(),0.0) << std::endl;
  //std::cout << "value trace bias: " << value_trace[-1] << std::endl;
  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller"); 
  std::fill(policy_update.begin(), policy_update.end(), 0.0f); 
  for (CSpace::TMapPerType::iterator it = cEntities.begin();
		  it != cEntities.end(); ++it) {
	  CControllableEntity *pcEntity =
		  any_cast<CControllableEntity *>(it->second);	    
	  try {
		  CEpuckNNController& cController =
			  dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
		  std::vector<float> update = cController.GetPolicyEligibilityTrace();
		  for (int i = 0; i < size_policy_net; ++i) {			            
			  policy_update[i] += update[i];			
    		  }
	  } catch (std::exception &ex) {
		  LOGERR << "Error while updating policy trace: " << ex.what() << std::endl;
	  }
  }
  // std::cout << "accumulate swarm policy trace: " << accumulate(policy_update.begin(),policy_update.end(),0.0) << std::endl;
  // actor
  Data policy_data {delta, policy_update};   
  std::string serialized_data = serialize(policy_data);
  zmq::message_t message(serialized_data.size());
  memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
  m_socket_actor.send(message, zmq::send_flags::dontwait);
  // critic
  Data value_data {delta, value_trace};   
  serialized_data = serialize(value_data);
  message.rebuild(serialized_data.size());
  memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
  m_socket_critic.send(message, zmq::send_flags::dontwait);  
  fTimeStep += 1;
  // std::cout << "step: " << fTimeStep << std::endl;
  // std::cout << "Poststep over\n";
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

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");

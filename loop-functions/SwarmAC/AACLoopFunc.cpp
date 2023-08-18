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

  delta = 0;
  policy_trace.resize(size_policy_net);
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
  value_trace.resize(size_value_net);
  std::fill(value_trace.begin(), value_trace.end(), 0.0f);

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

  m_context = zmq::context_t(1);
  m_socket_actor = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_actor.connect("tcp://localhost:5555");

  m_socket_critic = zmq::socket_t(m_context, ZMQ_PUSH);
  m_socket_critic.connect("tcp://localhost:5556");

  // Critic network

  // Initialise traces and delta
  delta = 0;
  policy_trace.resize(size_policy_net);
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
  value_trace.resize(size_value_net);
  std::fill(value_trace.begin(), value_trace.end(), 0.0f);
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
  //std::cout << "Prestep\n";
  // Observe state S
  /* Check position for each agent. 
   * If robot is in cell i --> 1
   * else 0
   */
  at::Tensor grid = torch::zeros({50, 50});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    
    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));

    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;
  }
  // Flatten the 2D tensor to 1D for net input
  state = grid.view({-1}).clone();
  state.set_requires_grad(true);

  // Load up-to-date value network
  std::vector<float> critic;
  std::vector<float> fc_input_weights;
  std::vector<float> fc_input_bias;
  std::vector<float> fc_output_weights;
  std::vector<float> fc_output_bias;

  const int fc_input_weights_count = input_size * hidden_size;  // input_features * output_features
  
  m_value->mutex.lock();
  for (auto w : m_value->vec) {
      critic.push_back(w);
      if (fc_input_weights.size() < fc_input_weights_count) {
          fc_input_weights.push_back(w);
      } else if (fc_input_weights.size() == fc_input_weights_count && fc_input_bias.size() < hidden_size) {
          fc_input_bias.push_back(w);
      } else {
          fc_output_weights.push_back(w);
          if (fc_output_weights.size() == hidden_size) {
              fc_output_bias.push_back(w);
          }
      }
  }
  m_value->mutex.unlock();
  //std::cout << "accumulate of value weights: " << accumulate(critic.begin(),critic.end(),0.0) << std::endl;
  // Update critic_net weights and bias
  for (auto& p : critic_net.named_parameters()) {
      if (p.key() == "fc_input.weight") {
          torch::Tensor new_weight_tensor = torch::from_blob(fc_input_weights.data(), {hidden_size, input_size});
          p.value().data().copy_(new_weight_tensor);
      } else if (p.key() == "fc_input.bias") {
          torch::Tensor new_bias_tensor = torch::from_blob(fc_input_bias.data(), {hidden_size});
          p.value().data().copy_(new_bias_tensor);
      } else if (p.key() == "fc_output.weight") {
          torch::Tensor new_weight_tensor = torch::from_blob(fc_output_weights.data(), {output_size, hidden_size});
          p.value().data().copy_(new_weight_tensor);
      } else if (p.key() == "fc_output.bias") {
          torch::Tensor new_bias_tensor = torch::from_blob(fc_output_bias.data(), {output_size});
          p.value().data().copy_(new_bias_tensor);
      }
  }
  // Load up-to-date policy network to each robot
  std::vector<float> actor;
  //std::cout << "Reading policy global shared vector\n";
  m_policy->mutex.lock();
  // 96 weights (24 * 4) + 4 bias for Dandelion
  // 288 weights (24 * 12) + 6 bias for Daisy
  //std::cout << "size global policy: " << m_policy->vec.size() << std::endl;
  for(auto w : m_policy->vec){
    actor.push_back(w);
    //std::cout << w << ", ";
  }
  m_policy->mutex.unlock();
  //std::cout << "accumulate of policy weights: " << accumulate(actor.begin(),actor.end(),0.0) << std::endl;
  //std::cout <<" actor\n";
  //std::cout << "global policy size: " << actor.size() << std::endl;
   
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
      cController.LoadNetwork(actor);
    } catch (std::exception &ex) {
      LOGERR << "Error while setting network: " << ex.what() << std::endl;
    }
  }
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  at::Tensor grid = torch::zeros({50, 50});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  int reward = 0;
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

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
  }

  //if(m_unScoreSpot1 > 0){
  //std::cout << "score: " << m_unScoreSpot1 << std::endl;	  
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
  //std::cout << "State:\n";
  //print_grid(grid);
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
  // Flatten the 2D tensor to 1D for net input
  state_prime = grid.view({-1}).clone();

  // Compute delta with Critic predictions
  int gamma = 1;
  torch::Tensor v_state = critic_net.forward(state);
  torch::Tensor v_state_prime = critic_net.forward(state_prime);
  delta = reward + gamma * v_state_prime[0].item<float>() - v_state[0].item<float>();
  /*std::cout << "v(s) = " << v_state[0].item<float>() << std::endl;
  std::cout << "v(s') = " << v_state_prime[0].item<float>() << std::endl;
  std::cout << "reward = " << reward << std::endl;
  std::cout << "delta = " << delta << std::endl;
  */
  // Compute value trace
  // Zero out the gradients
  critic_net.zero_grad();
  // Compute the gradient by back-propagation
  v_state.backward();

  auto params = critic_net.named_parameters();
  auto fc_input_weight_grad = params["fc_input.weight"].grad();
  auto fc_input_bias_grad = params["fc_input.bias"].grad();
  auto fc_output_weight_grad = params["fc_output.weight"].grad();
  auto fc_output_bias_grad = params["fc_output.bias"].grad();

  // Flatten the gradient tensors
  fc_input_weight_grad = fc_input_weight_grad.view({-1});
  fc_output_weight_grad = fc_output_weight_grad.view({-1});
  fc_input_bias_grad = fc_input_bias_grad.view({-1});
  fc_output_bias_grad = fc_output_bias_grad.view({-1});

  float* fc_input_weight_data = fc_input_weight_grad.data_ptr<float>();
  float* fc_output_weight_data = fc_output_weight_grad.data_ptr<float>();
  float* fc_input_bias_data = fc_input_bias_grad.data_ptr<float>();
  float* fc_output_bias_data = fc_output_bias_grad.data_ptr<float>();

  float lambda_critic = 0.9;

  // Update the value_trace based on the gradients for fc_input (weights and biases)
  for (int i = 0; i < input_size * hidden_size; ++i) {
      value_trace[i] = lambda_critic * value_trace[i] + fc_input_weight_data[i];
  }
  for (int i = input_size * hidden_size; i < (input_size * hidden_size + hidden_size); ++i) {
      value_trace[i] = lambda_critic * value_trace[i] + fc_input_bias_data[i - input_size * hidden_size];
  }


  // Update the value_trace based on the gradients for fc_output (weights and biases)
  for (int i = input_size * hidden_size + hidden_size; i < (input_size * hidden_size + hidden_size + hidden_size * output_size); ++i) {
      value_trace[i] = lambda_critic * value_trace[i] + fc_output_weight_data[i - input_size * hidden_size - hidden_size];
  }

  value_trace[input_size * hidden_size + hidden_size + hidden_size * output_size] = lambda_critic * value_trace[input_size * hidden_size + hidden_size + hidden_size * output_size] + fc_output_bias_data[0];

  //std::cout << "accumulate of value input weights: " << params["fc_input.weight"].sum() << std::endl;
  //std::cout << "accumulate of value output weights: " << params["fc_output.weight"].sum() << std::endl;
  //std::cout << "accumulate of value trace: " << accumulate(value_trace.begin(),value_trace.end(),0.0) << std::endl;
  //std::cout << "value trace bias: " << value_trace[-1] << std::endl;
  //std::cout <<" policy_tarce\n";
  
  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller"); 
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f); 
  for (CSpace::TMapPerType::iterator it = cEntities.begin();
		  it != cEntities.end(); ++it) {
	  CControllableEntity *pcEntity =
		  any_cast<CControllableEntity *>(it->second);	    
	  try {
		  CEpuckNNController& cController =
			  dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
		  std::vector<float> trace = cController.GetPolicyEligibilityTrace();
		  for (int i = 0; i < size_policy_net; ++i) {			            
			  policy_trace[i] += trace[i];			
    		  }
	  } catch (std::exception &ex) {
		  LOGERR << "Error while updating policy trace: " << ex.what() << std::endl;
	  }
  }
  //std::cout << "accumulate swarm policy trace: " << accumulate(policy_trace.begin(),policy_trace.end(),0.0) << std::endl;
  // actor
  Data policy_data {delta, policy_trace};   
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
}

/****************************************/
/****************************************/

void AACLoopFunction::PostExperiment() {
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

/**
  * @file <loop-functions/AggregationTwoSpotsLoopFunc.cpp>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @license MIT License
  */

#include "Homing.h"

/****************************************/
/****************************************/

Homing::Homing() {
  m_fRadius = 0.3;
  m_cCoordSpot1 = CVector2(0,-0.7);
  m_unScoreSpot1 = 0;
  m_fObjectiveFunction = 0;
  m_fHeightNest = 0.6;
}

/****************************************/
/****************************************/

Homing::Homing(const Homing& orig) {}

/****************************************/
/****************************************/

void Homing::Init(TConfigurationNode& t_tree) {
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
    policy_trace.resize(50);
    std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
    value_trace.resize(2501);
    std::fill(value_trace.begin(), value_trace.end(), 0.0f);
}

/****************************************/
/****************************************/


Homing::~Homing() {}

/****************************************/
/****************************************/

void Homing::Destroy() {}

/****************************************/
/****************************************/

argos::CColor Homing::GetFloorColor(const argos::CVector2& c_position_on_plane) {
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  Real d = (m_cCoordSpot1 - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::BLACK;
  }

  return CColor::GRAY50;
}


/****************************************/
/****************************************/

void Homing::Reset() {
  m_fObjectiveFunction = 0;
  m_unScoreSpot1 = 0;

  delta = 0;
  policy_trace.resize(50);
  std::fill(policy_trace.begin(), policy_trace.end(), 0.0f);
  value_trace.resize(2501);
  std::fill(value_trace.begin(), value_trace.end(), 0.0f);

  CoreLoopFunctions::Reset();
}

/****************************************/
/****************************************/

void Homing::PreStep() {
  // Observe state S
  /* Check position for each agent. 
   * If robot is in cell i --> 1
   * else 0
   */
  //Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(50,50);
  at::Tensor grid = torch::zeros({50, 50});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    
    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((cEpuckPosition.GetY() + 1.231) / 2.462 * 49));

    // Set the grid cell that corresponds to the epuck's position to 1
    //grid(grid_y, grid_x) = 1;
    grid[grid_y][grid_x] = 1;
  }
  //std::cout << "state:\n" << grid << std::endl;
  // convert matrix to vector for net input
  //state = Eigen::Map<Eigen::VectorXd>(grid.data(), grid.size());
  // Flatten the 2D tensor to 1D for net input
  state = grid.view({-1}).clone();

  // Load up-to-date value network
  //std::vector<MiniDNN::Scalar> critic;
  std::vector<float> critic;
  std::vector<float> critic_weights;
  std::vector<float> critic_bias;

  // Assuming dimensions of weights {output_features, input_features} for fc1 layer
  m_value->mutex.lock();
  for(auto w : m_value->vec){
      critic.push_back(w);
      if(critic_weights.size() < 2500){
        critic_weights.push_back(w);
      }else if (critic_weights.size() == 2500)
      {
        critic_bias.push_back(w);
      }
    //std::cout << w << ", ";
  }
  m_value->mutex.unlock();
  //std::cout << "Critic parameters: " << critic_net.parameters() << std::endl;
  // Upadate critic weights and bias
  for (auto& p : critic_net.named_parameters()) {
      if (p.key() == "fc.weight") {
        //std::cout << p.key() << ": " << p.value() << std::endl;
        //std::cout << "Critic: " << critic << std::endl;
        torch::Tensor new_weight_tensor = torch::from_blob(critic_weights.data(), {1, 2500});
        p.value().data().copy_(new_weight_tensor);
      } else if (p.key() == "fc.bias") {
        //std::cout << p.key() << "_critic: " << p.value() << std::endl;
        torch::Tensor new_bias_tensor = torch::from_blob(critic_bias.data(), {1});
        p.value().data().copy_(new_bias_tensor);
      }
  }

  //auto layer = const_cast<MiniDNN::Layer *>(m_criticNet.get_layers()[0]);
  //layer->set_parameters(critic);
  //std::cout <<" critic\n";

  // Load up-to-date policy network to each robot
  //std::vector<MiniDNN::Scalar> actor;
  std::vector<float> actor;
  m_policy->mutex.lock();
  for(auto w : m_policy->vec){
    actor.push_back(w);
    //std::cout << w << ", ";
  }
  m_policy->mutex.unlock();
  //std::cout <<" actor\n";
  
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
      cController.LoadNetwork(actor);
    } catch (std::exception &ex) {
      LOGERR << "Error while setting network: " << ex.what() << std::endl;
    }
  }
}

/****************************************/
/****************************************/

void Homing::PostStep() {
  //Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(50,50);
  at::Tensor grid = torch::zeros({50, 50});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

    Real fDistanceSpot1 = (m_cCoordSpot1 - cEpuckPosition).Length();
    if (fDistanceSpot1 <= m_fRadius) {
      //m_unScoreSpot1 += 1;
      m_fObjectiveFunction += 1;    // Reward    
    }

    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((cEpuckPosition.GetY() + 1.231) / 2.462 * 49));

    // Set the grid cell that corresponds to the epuck's position to 1
    //grid(grid_y, grid_x) = 1;
    grid[grid_y][grid_x] = 1;
  }
  //m_fObjectiveFunction = m_unScoreSpot1/(Real) m_unNumberRobots;
  //LOG << "Score = " << m_fObjectiveFunction << std::endl;

  // Observe state S'
  /* Check position for each agent. 
   * If robot is in grid --> 1
   * else 0
   */
  //state_prime = Eigen::Map<Eigen::VectorXd>(grid.data(), grid.size());
  // Flatten the 2D tensor to 1D for net input
  state_prime = grid.view({-1}).clone();

  // Compute delta with Critic predictions
  //Matrix v_state = m_criticNet.predict(state);
  torch::Tensor v_state = critic_net.forward(state);
  //std::cout << "v(s) = " << v_state[0].item<float>() << std::endl;
  //Matrix v_state_prime = m_criticNet.predict(state_prime);
  torch::Tensor v_state_prime = critic_net.forward(state_prime);
  //std::cout << "v(s') = " << v_state_prime[0].item<float>() << std::endl;
  //delta = m_fObjectiveFunction + v_state(0) - v_state_prime(0);
  delta = m_fObjectiveFunction + v_state[0].item<float>() - v_state_prime[0].item<float>();
  //std::cout << "delta = " << delta << std::endl;

  // Compute policy trace
  // TODO: add gradiant
  for(auto& element : policy_trace) {
    element *= 0.01;     //lambda
    //std::cout << element << ", ";
  }
  //std::cout <<" policy_tarce\n";

  // Compute value trace
  // TODO: add gradiant
  // Initialize critic_trace as a std::vector<torch::Tensor>
  // Zero out the gradients
  critic_net.zero_grad();
  // Compute the gradient by back-propagation
  v_state.backward();
  std::cout << "value trace: " << value_trace << std::endl;
  // if critic_trace is not empty, update it
  int i = 0;
  for (auto& p : critic_net.parameters()) {
	  torch::Tensor value_trace_tensor = torch::tensor(value_trace[i]);
	  value_trace_tensor = value_trace_tensor * 0.1 + p.grad();
	  // Convert the tensor back to a float and update the ith element of value_trace
	  //value_trace[i] = value_trace_tensor.item<float>();
	  auto value_trace_tensor_accessor = value_trace_tensor.accessor<float,1>();

	  for(int i = 0; i < value_trace_tensor_accessor.size(0); i++)
	  {
		  std::cout << "here\n";    
		  value_trace[i] = value_trace_tensor_accessor[i];
		  std::cout << "there\n";
	  }

	  ++i;
  }
  // for(auto& element : value_trace) {
  //   element *= 0.01;     //lambda
  //   //std::cout << element << ", ";
  // }
  //std::cout <<" value_tarce\n";

  // send message to manager
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

void Homing::PostExperiment() {
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
}

/****************************************/
/****************************************/

Real Homing::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

CVector3 Homing::GetRandomPosition() {
  Real temp, a, b, fPosX, fPosY;
  bool bPlaced = false;
  do {
      a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
      b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
      // If b < a, swap them
      if (b < a) {
        temp = a;
        a = b;
        b = temp;
      }
      fPosX = b * m_fDistributionRadius * cos(2 * CRadians::PI.GetValue() * (a/b));
      fPosY = b * m_fDistributionRadius * sin(2 * CRadians::PI.GetValue() * (a/b));

      if (fPosY >= m_fHeightNest) {
        bPlaced = true;
      }
  } while(!bPlaced);
  return CVector3(fPosX, fPosY, 0);
}

/****************************************/
/****************************************/

// Function to compute the log-PDF of the Beta distribution
double computeBetaLogPDF(double alpha, double beta, double x) {
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

void Homing::update_actor(torch::nn::Module& actor_net, torch::Tensor& behavior_probs, 
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

REGISTER_LOOP_FUNCTIONS(Homing, "homing");

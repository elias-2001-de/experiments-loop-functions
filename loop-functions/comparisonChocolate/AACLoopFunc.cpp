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

void AACLoopFunction::Destroy() {
  //std::cout << "Before Destroy" << "." << std::endl;
  agents.clear();
  //RemoveArena();
}

/****************************************/
/****************************************/

void AACLoopFunction::Reset() {
  m_fObjectiveFunction = 0;
  fTimeStep = 0;

  for (MADDPGLoopFunction::Agent* a : agents){
    a->policy_param.resize(size_policy_net);
    std::fill(a->policy_param.begin(), a->policy_param.end(), 0.0f);
    a->value_param.resize(size_value_net);
    std::fill(a->value_param.begin(), a->value_param.end(), 0.0f);
    a->optimizer_critic->zero_grad();
    a->optimizer_actor->zero_grad();
  }

  MADDPGLoopFunction::Reset();
}

/****************************************/
/****************************************/

void AACLoopFunction::Init(TConfigurationNode& t_tree) {

  int actor_hidden_dim;
  int actor_num_hidden_layers;
  int actor_output_dim;
  float lambda_actor;
  float lambda_critic;
  int critic_hidden_dim;
  int critic_num_hidden_layers;
  int critic_output_dim;


  MADDPGLoopFunction::Init(t_tree);
  std::cout << "Before Init loop function" << "." << std::endl;

  //std::cout << "Size agents" << agents.size() << std::endl;

  TConfigurationNode cParametersNode;
  try {
     cParametersNode = GetNode(t_tree, "params");
  } catch(std::exception e) {
  }
  GetNodeAttribute(cParametersNode, "number_robots", nb_robots);
  GetNodeAttribute(cParametersNode, "buffer_size", max_buffer_size);
  GetNodeAttribute(cParametersNode, "batch_size", batch_size);
  GetNodeAttribute(cParametersNode, "batch_step", batch_step);
  GetNodeAttribute(cParametersNode, "gamma", gamma);
  GetNodeAttribute(cParametersNode, "tau", tau);

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", critic_input_dim);
  GetNodeAttribute(criticParameters, "hidden_dim", critic_hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", critic_num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", critic_output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);

  TConfigurationNode actorParameters;
  actorParameters = GetNode(t_tree, "actor");

  GetNodeAttribute(actorParameters, "input_dim", actor_input_dim);
  GetNodeAttribute(actorParameters, "hidden_dim", actor_hidden_dim);
  GetNodeAttribute(actorParameters, "num_hidden_layers", actor_num_hidden_layers);
  GetNodeAttribute(actorParameters, "output_dim", actor_output_dim);
  GetNodeAttribute(actorParameters, "lambda_actor", lambda_actor);

  if(actor_num_hidden_layers>0){
    size_policy_net = (actor_input_dim*actor_hidden_dim+actor_hidden_dim) + (actor_num_hidden_layers-1)*(actor_hidden_dim*actor_hidden_dim+actor_hidden_dim) + (actor_hidden_dim*actor_output_dim+actor_output_dim);
  }else{
    size_policy_net = actor_input_dim*actor_output_dim + actor_output_dim;
  }

  //std::cout << "Critic hidden layers" << critic_num_hidden_layers << std::endl;
  if(critic_num_hidden_layers>0){
    size_value_net = (critic_input_dim*critic_hidden_dim+critic_hidden_dim) + (critic_num_hidden_layers-1)*(critic_hidden_dim*critic_hidden_dim+critic_hidden_dim) + (critic_hidden_dim*critic_output_dim+critic_output_dim);
  }else{
    size_value_net = critic_input_dim*critic_output_dim + critic_output_dim;
  }

  for (int r = 0; r < nb_robots; r++) {
    MADDPGLoopFunction::Agent* agent = new MADDPGLoopFunction::Agent(critic_input_dim, critic_hidden_dim, critic_num_hidden_layers, critic_output_dim, actor_input_dim, actor_hidden_dim, actor_num_hidden_layers, actor_output_dim, lambda_actor, lambda_critic, size_value_net, size_policy_net);
    agents.push_back(agent);
  }

  argos::TConfigurationNode& parentNode = *dynamic_cast<argos::TConfigurationNode*>(t_tree.Parent());
  TConfigurationNode frameworkNode;
  TConfigurationNode experimentNode;
  frameworkNode = GetNode(parentNode, "framework");
  experimentNode = GetNode(frameworkNode, "experiment");
  GetNodeAttribute(experimentNode, "length", mission_lengh);

  fTimeStep = 0;
  fTimeBatch = 0;
  fTimeStepTraining = 0;

  SetControllerEpuckAgent();
}

void AACLoopFunction::SetControllerEpuckAgent() {
  //std::cout << "Before SetControllerEpuckAgent()" << "." << std::endl;
  int i = 0;

  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    agents.at(i)->pcEpuck = pcEpuck;
    agents.at(i)->critic.to(device);
    agents.at(i)->actor.to(device);
    i += 1;
  }

  i = 0;
  CSpace::TMapPerType cEntities = GetSpace().GetEntitiesByType("controller");
  for (CSpace::TMapPerType::iterator it = cEntities.begin();
       it != cEntities.end(); ++it) {
    CControllableEntity *pcEntity = any_cast<CControllableEntity *>(it->second);
    //std::cout << "Before per agent" << "." << std::endl;
    CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(pcEntity->GetController());
    cController.SetNetworkAndOptimizer(agents.at(i)->actor, agents.at(i)->optimizer_actor);
    cController.SetTargetNetwork(agents.at(i)->actor);
    agents.at(i)->pcEntity = pcEntity;
    i += 1;
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

  // d = (m_cCoordWhiteSpot - vCurrentPoint).Length();
  // if (d <= m_fRadius) {
  //   return CColor::WHITE;
  // }

  return CColor::GRAY50;
}

/****************************************/
/****************************************/

void AACLoopFunction::PreStep() {
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  // Assuming that an action is composed of two floats (one for each wheel)
  //std::cout << "Before poststep" << "." << std::endl;
  const int doubleNbRobots = 2*nb_robots;
  std::vector<MADDPGLoopFunction::Transition*> sample;
  sample.reserve(batch_size);  
  torch::Tensor target_actions = torch::empty({doubleNbRobots});
  torch::Tensor actions = torch::empty({doubleNbRobots});
  torch::Tensor pos = torch::empty({4*nb_robots});
  std::vector<torch::Tensor> actions_trans;
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  CRadians cRoll, cPitch, cYaw;
  std::vector<int> rewards(nb_robots, 0);
  std::vector<torch::Tensor> next_obs;
  int pos_index = 0;
  int i = 0;
  //std::cout << "Before for agent" << "." << std::endl;  
  for (MADDPGLoopFunction::Agent* a : agents){
    cEpuckPosition.Set(a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    cEpuckOrientation = a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);

    pos = pos.index_put_({pos_index}, cEpuckPosition.GetX());
    pos = pos.index_put_({pos_index+1}, cEpuckPosition.GetY());
    pos = pos.index_put_({pos_index+2}, std::cos(cYaw.GetValue()));
    pos = pos.index_put_({pos_index+3}, std::sin(cYaw.GetValue()));
    pos_index += 4;


    Real fDistanceSpot = (m_cCoordBlackSpot - cEpuckPosition).Length();
    if (fDistanceSpot <= m_fRadius) {
      m_fObjectiveFunction += 1;
      rewards[i] = 1;
    }else{
      rewards[i] = 0;
    }

    // Get observations that the robots have after the step
    CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(a->pcEntity->GetController());
    if (fTimeStep == 0){
      obs.push_back(cController.Observe());
      next_obs.push_back(cController.Observe());
    }else{
      next_obs.push_back(cController.Observe());
      // Get action that the robot did through actor network
      actions_trans.push_back(a->actor.GetAction());
    }
  }

  //std::cout << "Time step " << fTimeStep << std::endl; 
  if (fTimeStep == 0){
    //std::cout << "New state" << std::endl; 
    state = pos.clone();
    state_prime = pos.clone();
  }else {
    // Next_state after the step
    state_prime = pos.clone();
    // Need to add this transition to the buffer
    //std::cout << "Size state in transition " << state.sizes() << std::endl;
    MADDPGLoopFunction::Transition* transition = new MADDPGLoopFunction::Transition(state, obs, actions_trans, rewards, state_prime, next_obs);
    buffer.push_back(transition); 
    if (buffer.size() > max_buffer_size){
      //std::cout << "Elem buffer " << buffer.at(0) << std::endl; 
      delete buffer.at(0);
      buffer.at(0) = nullptr;
      //std::cout << "Elem buffer " << buffer.at(0) << std::endl; 
      buffer.erase(std::remove(buffer.begin(), buffer.end(), nullptr), buffer.end());
    }
  }

  // Beginning of the learning loop if we are in training mode and there are enough samples in the buffer
  int buffer_size = static_cast<int>(buffer.size());
  //std::cout << "buffer size" << buffer_size << std::endl;
  //std::cout << "batch size" << batch_size << std::endl;
  if(training && buffer_size >= batch_size){ 
    if(fTimeBatch == 0){
      //std::cout << "In if training" << "." << std::endl;
      //state = state.to(device);
      //state_prime = state_prime.to(device);
      //actions = actions.to(device);
      //target_actions = target_actions.to(device);
      
      // Update on the networks for each agent
      //std::cout << "Before for agents_size" << "." << std::endl;  
      for (int a=0; a < nb_robots; a++){
          //First, need to sample a certain number of transitions from the buffer
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<> dis(0, buffer_size - 1);
          //TODO: chnager le 3
          for (int i = 0; i < batch_size; ++i) {
              int randomIndex = dis(gen);
              sample.push_back(buffer.at(randomIndex));
          }

          //std::cout<< "Critic model update" << std::endl;
          std::vector<torch::Tensor> states_vec, actions_vec, rewards_vec, next_states_vec, target_actions_vec;
          for (int j=0; j<sample.size(); j++){
              states_vec.push_back(sample.at(j)->state);
              actions_vec.push_back(torch::cat(sample.at(j)->actions, 0)); // Concatenate the action tensors
              rewards_vec.push_back(torch::tensor(sample.at(j)->rewards.at(a), torch::kFloat32).unsqueeze(0)); // Convert reward to tensor
              next_states_vec.push_back(sample.at(j)->state_prime);

              // Construct target_actions for each sample
              std::vector<torch::Tensor> target_actions_to_add;
              for (int i = 0; i < nb_robots; i++) {
                  CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(i)->pcEntity->GetController());
                  std::vector<torch::Tensor> vec_actions = cController.TargetAction(sample.at(j)->next_obs.at(i));
                  for (auto& action : vec_actions) {
                      target_actions_to_add.push_back(action.unsqueeze(0)); // Add a new dimension to the action tensor
                  }
              }
              target_actions_vec.push_back(torch::cat(target_actions_to_add, 0)); // Concatenate the target action tensors            
          }
          // Convert vectors to tensors 
          torch::Tensor states_batch = torch::stack(states_vec, 0);
          torch::Tensor actions_batch = torch::stack(actions_vec, 0); // This will now have actions from all robots
          torch::Tensor rewards_batch = torch::stack(rewards_vec, 0);
          torch::Tensor next_states_batch = torch::stack(next_states_vec, 0);
          torch::Tensor target_actions_batch = torch::stack(target_actions_vec, 0); // This will now have target actions for all samples
          
          // Calculate critic value without detaching
          torch::Tensor input_critic = torch::cat({states_batch, actions_batch}, 1);
          torch::Tensor critic_value = agents.at(a)->critic.forward(input_critic)[0];

          // Calculate y for each transition in the sample
          torch::Tensor input_target_critic = torch::cat({next_states_batch, target_actions_batch}, 1);
          input_target_critic = input_target_critic.toType(torch::kFloat32); // Convert to float
          torch::Tensor y = rewards_batch + gamma * agents.at(a)->target_critic.forward(input_target_critic)[0];

          // Calculate loss
          torch::Tensor squared_diff = torch::pow(y - critic_value, 2);
          torch::Tensor mse = torch::mean(squared_diff);

          // Calculate gradient of loss
          agents.at(a)->optimizer_critic->zero_grad();
          mse.backward();
          agents.at(a)->optimizer_critic->step();


          //std::cout << "Policy model update" << std::endl;
          torch::Tensor value_policy = torch::empty({sample.size()}); 
          for (int j = 0; j < sample.size(); j++) {
              int index = 0;
              for (int i = 0; i < nb_robots; i++) {
                  if (a == i) {
                      CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(a)->pcEntity->GetController());
                      std::vector<torch::Tensor> actions_to_add = cController.Action(sample.at(j)->obs.at(a));
                      actions = actions.index_put_({index}, actions_to_add.at(0));
                      actions = actions.index_put_({index+1}, actions_to_add.at(1));
                  }
                  else {
                      actions = actions.index_put_({index}, sample.at(j)->actions.at(i)[0]);
                      actions = actions.index_put_({index+1}, sample.at(j)->actions.at(i)[1]);                    
                  }
                  index += 2;
              }
              torch::Tensor input_critic = torch::cat({sample.at(j)->state, actions});

              // Use of the critic network of the current agent
              value_policy= value_policy.index_put_({j}, agents.at(a)->critic.forward(input_critic)[0]);
          }

          //Policy loss
          torch::Tensor actor_loss = -torch::mean(value_policy);

          agents.at(a)->optimizer_actor->zero_grad();
          actor_loss.backward();
          agents.at(a)->optimizer_actor->step();
      }

      // Updqte of the target networks
      //std::cout << "Here" << std::endl;
      if (fTimeStepTraining % 100 == 0){
        std::cout << "Here" << std::endl;
        for (int a=0; a<nb_robots; a++){
            int target_params_size = static_cast<int>(agents.at(a)->target_critic.parameters().size());
            for (int p = 0; p < target_params_size; p++) {
                agents.at(a)->target_critic.parameters().at(p) = torch::mul(agents.at(a)->critic.parameters().at(p), tau) + torch::mul(agents.at(a)->target_critic.parameters().at(p), (1 - tau));
                CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(a)->pcEntity->GetController());
                CEpuckNNController::Actor_Net* target_network = cController.GetTargetNetwork();
                target_network->parameters().at(p) = torch::mul(agents.at(a)->actor.parameters().at(p), tau) + torch::mul(target_network->parameters().at(p), (1 - tau));
            }
        }
      }
    std::cout << "step: " << fTimeStep << std::endl;
    }else{
      fTimeBatch += 1;
      if (fTimeBatch == batch_step){
        fTimeBatch = 0;
      }
    }
    fTimeStepTraining += 1;
  }
  fTimeStep += 1;
  obs = next_obs;
  state = state_prime;
}

/****************************************/
/****************************************/

void AACLoopFunction::PostExperiment() {
  LOG << "Time = " << fTimeStep << std::endl;
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
  std::cout << "NB transitions in buffer " << buffer.size() << std::endl;
}

/****************************************/
/****************************************/

Real AACLoopFunction::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

CVector3 AACLoopFunction::GetRandomPosition() {

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

void AACLoopFunction::update_actor(torch::nn::Module actor_net, torch::Tensor& behavior_probs, 
                  torch::Tensor& beta_parameters, int chosen_behavior, 
                  double chosen_parameter, double td_error, 
                  torch::optim::Adam* optimizer) {

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
    optimizer->zero_grad();

    // Compute gradients w.r.t. actor network parameters (weight and biases)
    loss.backward();

    // Perform one step of the optimization (gradient descent)
    optimizer->step();
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

void AACLoopFunction::SetVectorAgents(std::vector<MADDPGLoopFunction::Agent*> vectorAgents) {
    agents = vectorAgents;
}

void AACLoopFunction::SetBuffer(std::vector<MADDPGLoopFunction::Transition*> vectorTransitions) {
    buffer = vectorTransitions;
}

void AACLoopFunction::SetSizeValueNet(int value) {
    size_value_net = value;
}

void AACLoopFunction::SetSizePolicyNet(int policy) {
    size_policy_net = policy;
}

void AACLoopFunction::RemoveArena() {
    std::ostringstream id;
    id << "arena";
    RemoveEntity(id.str().c_str());
}

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");
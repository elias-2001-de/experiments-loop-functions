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
  critic_losses = 0;
  actor_losses = 0;
  entropies = 0.0;
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
  //std::cout << "In the reset part" << std::endl;
  m_fObjectiveFunction = 0;
  fTimeStep = 0;

  for (MADDPGLoopFunction::Agent* a : agents){
    a->optimizer_critic->zero_grad();
    a->optimizer_actor->zero_grad();
  }

  critic_losses = 0.0;
  actor_losses = 0.0;
  entropies = 0.0;

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
  std::string device_to_put;
  int option_input;


  MADDPGLoopFunction::Init(t_tree);
  //std::cout << "Before Init loop function" << "." << std::endl;

  //std::cout << "Size agents" << agents.size() << std::endl;

  TConfigurationNode cParametersNode;
  try {
     cParametersNode = GetNode(t_tree, "params");
  } catch(std::exception e) {
  }
  GetNodeAttribute(cParametersNode, "number_robots", nb_robots);
  GetNodeAttribute(cParametersNode, "buffer_size", max_buffer_size);
  GetNodeAttribute(cParametersNode, "batch_size", batch_size);
  GetNodeAttribute(cParametersNode, "batch_begin", batch_begin);
  GetNodeAttribute(cParametersNode, "batch_step", batch_step);
  GetNodeAttribute(cParametersNode, "update_target", update_target);
  GetNodeAttribute(cParametersNode, "gamma", gamma);
  GetNodeAttribute(cParametersNode, "tau", tau);
  GetNodeAttribute(cParametersNode, "device", device_to_put);

  TConfigurationNode criticParameters;
  criticParameters = GetNode(t_tree, "critic");
  
  GetNodeAttribute(criticParameters, "input_dim", critic_input_dim);
  GetNodeAttribute(criticParameters, "hidden_dim", critic_hidden_dim);
  GetNodeAttribute(criticParameters, "num_hidden_layers", critic_num_hidden_layers);
  GetNodeAttribute(criticParameters, "output_dim", critic_output_dim);
  GetNodeAttribute(criticParameters, "lambda_critic", lambda_critic);
  GetNodeAttribute(criticParameters, "option_input", option_input);

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
    agent->actor.SetInputActor(option_input);
    agent->target_actor.SetInputActor(option_input);
    agents.push_back(agent);
  }

  argos::TConfigurationNode& parentNode = *dynamic_cast<argos::TConfigurationNode*>(t_tree.Parent());
  TConfigurationNode frameworkNode;
  TConfigurationNode experimentNode;
  frameworkNode = GetNode(parentNode, "framework");
  experimentNode = GetNode(frameworkNode, "experiment");
  GetNodeAttribute(experimentNode, "length", mission_lengh);

  fTimeStep = 0;
  fTimeStepTraining = 0;

  SetDevice(device_to_put);

  SetControllerEpuckAgent();

  buffer.resize(max_buffer_size);
}

void AACLoopFunction::SetDevice(std::string device_to_put){
  std::cout << "CPU to do the calculations" << std::endl;
  if(device_to_put == "GPU"){
    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }
  }
}

void AACLoopFunction::SetControllerEpuckAgent() {
  //std::cout << "Before SetControllerEpuckAgent()" << "." << std::endl;
  int i = 0;

  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    std::cout << device << std::endl;
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    agents.at(i)->pcEpuck = pcEpuck;
    agents.at(i)->critic.to(device);
    agents.at(i)->actor.to(device);
    agents.at(i)->target_critic.to(device);
    agents.at(i)->target_actor.to(device);
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
    cController.SetTargetNetwork(agents.at(i)->target_actor);
    cController.SetDevice(device);
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

  d = (m_cCoordWhiteSpot - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::WHITE;
  }

  return CColor::GRAY50;
}

/****************************************/
/****************************************/

void AACLoopFunction::PreStep() {
  //std::cout << "Pre-Step" << std::endl;
  CVector2 cEpuckPosition(0,0);
  CQuaternion cEpuckOrientation;
  CRadians cRoll, cPitch, cYaw;
  for (MADDPGLoopFunction::Agent* a : agents){
    torch::Tensor pos = torch::empty({4});
    cEpuckPosition.Set(a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    cEpuckOrientation = a->pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Orientation;
    cEpuckOrientation.ToEulerAngles(cYaw, cPitch, cRoll);

    pos = pos.index_put_({0}, cEpuckPosition.GetX());
    pos = pos.index_put_({1}, cEpuckPosition.GetY());
    pos = pos.index_put_({2}, std::cos(cYaw.GetValue()));
    pos = pos.index_put_({3}, std::sin(cYaw.GetValue()));
    
    CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(a->pcEntity->GetController());

    cController.SetPosRobot(pos);
  }
}

/****************************************/
/****************************************/

void AACLoopFunction::PostStep() {
  
  // Create an ofstream object for output
  //torch::autograd::AnomalyMode::set_enabled(true);
  //std::ofstream outfile;
  //outfile.open("debug_losses.txt", std::ios_base::app); // append instead of overwrite
  // Assuming that an action is composed of two floats (one for each wheel)
  const int doubleNbRobots = 2*nb_robots;
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
    //std::cout << "Before dynamic cast" << fTimeStepTraining << std::endl;
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

  if (fTimeStep == 0){
    state = pos.clone().to(device);
    state_prime = pos.clone().to(device);
  }else {
    // Next_state after the step
    state_prime = pos.clone().to(device);
    // Need to add this transition to the buffer
    MADDPGLoopFunction::Transition* transition = new MADDPGLoopFunction::Transition(state, obs, actions_trans, rewards, state_prime);
    buffer.push(transition); 
    if (buffer.is_full()) {
      MADDPGLoopFunction::Transition* oldest_transition = buffer.pop();
      delete oldest_transition;
    }
  }

  // Beginning of the learning loop if we are in training mode and there are enough samples in the buffer
  //std::cout << "Before learning" << std::endl;
  int buffer_size = static_cast<int>(buffer.size());
  if(training && buffer_size >= batch_begin){ 
    if(fTimeStepTraining % batch_step == 0){ 
      std::vector<std::thread> threads;
      for (int a=0; a < nb_robots; a++){
          //First, need to sample a certain number of transitions from the buffer
          std::vector<MADDPGLoopFunction::Transition*> sample;
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<> dis(0, buffer_size - 1);
          //std::cout << "Before loop buffer" << std::endl;
          for (int i = 0; i < batch_size; ++i) {
              int randomIndex = dis(gen);
              sample.push_back(buffer.at(randomIndex));
          }
          //Update(sample, a);
          threads.push_back(std::thread(&AACLoopFunction::Update, this, sample, a));
      }

      for (auto& t : threads){
        t.join();
      }

      // Update of the target networks
      if (fTimeStepTraining % update_target == 0){
        //std::cout << "Target update" << std::endl;
        for (int a=0; a<nb_robots; a++){
            for (size_t i = 0; i < agents.at(a)->critic.parameters().size(); ++i) {
                // Get the parameters of the critic and target critic networks
                torch::Tensor critic_param = agents.at(a)->critic.parameters()[i];
                torch::Tensor target_critic_param = agents.at(a)->target_critic.parameters()[i];

                // Perform the soft update
                target_critic_param.data().copy_(tau * critic_param.data() + (1.0 - tau) * target_critic_param.data());
            }
            for (size_t i = 0; i < agents.at(a)->actor.parameters().size(); ++i) {
                // Get the parameters of the actor and target actor networks
                torch::Tensor actor_param = agents.at(a)->actor.parameters()[i];
                CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(a)->pcEntity->GetController());
                CEpuckNNController::Actor_Net* target_network = cController.GetTargetNetwork();
                torch::Tensor target_actor_param = target_network->parameters()[i];

                // Perform the soft update
                target_actor_param.data().copy_(tau * actor_param.data() + (1.0 - tau) * target_actor_param.data());
            }
        }
        
      }
    }
    fTimeStepTraining += 1;
  }
  fTimeStep += 1;
  obs = next_obs;
  state = state_prime;
  //std::cout << "Fin de post-step" << std::endl;
  //outfile.close();
  //auto end_time = std::chrono::high_resolution_clock::now();
  //auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  //auto now_in_time_t = std::chrono::system_clock::to_time_t(start_time);
  //std::cout << "Start time before post step: " << std::ctime(&now_in_time_t) << " milliseconds" << std::endl;
  //std::cout << "Elapsed time between pre and post: " << elapsed_time << " milliseconds" << std::endl;
  //start_time = std::chrono::high_resolution_clock::now();
  //now_in_time_t = std::chrono::system_clock::to_time_t(start_time);
  //std::cout << "Start time after post step: " << std::ctime(&now_in_time_t) << " milliseconds" << std::endl;
}

void AACLoopFunction::Update(std::vector<MADDPGLoopFunction::Transition*> sample, int a){
  //std::cout << "a value: " << a << std::endl;
  //std::cout << "sample: " << sample.size() << std::endl;
  //Critic model update
  //std::cout << "Before critic update" << std::endl;
  std::vector<torch::Tensor> states_vec, actions_vec, rewards_vec, next_states_vec, target_actions_vec;
  // Create a vector of tensors for each robot's observations
  std::vector<std::vector<torch::Tensor>> all_obs(nb_robots);
  for (int j=0; j<sample.size(); j++){
    states_vec.push_back(sample.at(j)->state);
    actions_vec.push_back(torch::cat(sample.at(j)->actions, 0)); // Concatenate the action tensors
    rewards_vec.push_back(torch::tensor(sample.at(j)->rewards.at(a), torch::kFloat32).unsqueeze(0)); // Convert reward to tensor
    next_states_vec.push_back(sample.at(j)->state_prime);

    for (int i = 0; i < nb_robots; i++) {
      all_obs[i].push_back(sample.at(j)->obs.at(i));
    }
  }

    // Construct target_actions for each sample
  for (int i = 0; i < nb_robots; i++) {
    CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(i)->pcEntity->GetController());
    torch::Tensor batched_obs = torch::stack(all_obs[i], 0); // Stack the observations
    torch::Tensor action = cController.TargetAction(batched_obs);
    //std::cout << "Sizes target_actions: " << action.sizes() << std::endl;
    target_actions_vec.push_back(action); // Add a new dimension to the action tensor
  //std::cout << "a value: " << a << std::endl;
    //std::cout << "target_actions_vec: " << target_actions_vec.size() << std::endl;
  }         
  // After constructing vectors for critic model update
  //outfile << "States vector size: " << states_vec.size() << std::endl;
  //outfile << "States vector: " << states_vec << std::endl;
  //outfile << "Actions vector size: " << actions_vec.size() << std::endl;
  //std::cout << "Actions vector: " << actions_vec << std::endl;
  // Convert vectors to tensors 
  torch::Tensor states_batch = torch::stack(states_vec, 0);
  torch::Tensor actions_batch = torch::stack(actions_vec, 0); // This will now have actions from all robots
  torch::Tensor rewards_batch = torch::stack(rewards_vec, 0).to(device);
  //outfile << "Rewards batch: " << rewards_batch << std::endl;
  torch::Tensor next_states_batch = torch::stack(next_states_vec, 0);
  torch::Tensor target_actions_batch = torch::cat(target_actions_vec, 1); // This will now have target actions for all samples
  //std::cout << "a value: " << a << std::endl;
  //std::cout << "target_actions_batch: " << target_actions_batch.sizes() << std::endl;
          
  // Calculate critic value without detaching
  torch::Tensor input_critic = torch::cat({states_batch, actions_batch}, 1);
  torch::Tensor critic_value = agents.at(a)->critic.forward(input_critic);
  //for (auto& layer : agents.at(a)->critic.hidden_layers) {
    //outfile << "Weights layer: " << layer->weight << std::endl;
  //}
  //outfile << "Weights last layer: " << agents.at(a)->critic.output_layer->weight << std::endl;
  //outfile << "Critic value: " << critic_value << std::endl;
  //outfile << "Wrong critic value: " << critic_value[0] << std::endl;

  // Calculate y for each transition in the sample
  torch::Tensor input_target_critic = torch::cat({next_states_batch, target_actions_batch}, 1);
  input_target_critic = input_target_critic.toType(torch::kFloat32); // Convert to float
  //torch::Tensor target_output = agents.at(a)->target_critic.forward(input_target_critic);
  //outfile << "Target Q-value: " << target_output << std::endl;
  //outfile << "Weights target critic network:\n" << agents.at(a)->target_critic.parameters() << std::endl;
  torch::Tensor y = rewards_batch + gamma * agents.at(a)->target_critic.forward(input_target_critic);
  //outfile << "Y value: " << y << std::endl;

  //outfile << "Q-value: " << critic_value << std::endl;

  // Calculate loss
  torch::Tensor squared_diff = torch::pow(y - critic_value, 2);
  torch::Tensor mse = torch::mean(squared_diff);
  //outfile << "Critic loss: " << mse << std::endl;

  critic_losses += mse.item<float>();


  // Calculate gradient of loss
  //outfile << "Weights before update:\n" << agents.at(a)->critic.parameters() << std::endl;

  agents.at(a)->optimizer_critic->zero_grad();
  mse.backward();
  agents.at(a)->optimizer_critic->step();

  //outfile << "Weights after update:\n" << agents.at(a)->critic.parameters() << std::endl;
  //CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(a)->pcEntity->GetController());
  //entropies += cController.GetPolicyEntropy();
  //outfile << "Weights target actor network:\n" << cController.GetTargetNetwork()->parameters() << std::endl;

  //std::cout << "Before policy update" << std::endl;
  // Policy model update
  std::vector<torch::Tensor> actions_batch_vec;
  int index = 0;
  for (int i = 0; i < nb_robots; i++) {
    if (a == i) {
      CEpuckNNController& cController = dynamic_cast<CEpuckNNController&>(agents.at(a)->pcEntity->GetController());
      torch::Tensor batched_obs = torch::stack(all_obs[a], 0);
      //std::cout << "Size obs: " << sample.at(j)->obs.at(a).sizes() << std::endl;
      torch::Tensor actions_to_add = cController.Action(batched_obs);
      actions_batch_vec.push_back(actions_to_add);
    }else {
      torch::Tensor sliced1 = actions_batch.select(1,index);
      torch::Tensor sliced2 = actions_batch.select(1,index+1);
      torch::Tensor result = torch::cat({sliced1.unsqueeze(1), sliced2.unsqueeze(1)}, 1);
      actions_batch_vec.push_back(result);   
    }
    index += 2;
  }
  //outfile << "Actions batch for policy update size: " << actions_batch_pupdate.size(0) << std::endl;
  //std::cout << "After loop" << std::endl;
  torch::Tensor actions_batch_pupdate = torch::cat(actions_batch_vec, 1);
  //outfile << "Actions batch for policy update size: " << actions_batch_pupdate.size(0) << std::endl;
  torch::Tensor input_critic_pupdate = torch::cat({states_batch, actions_batch_pupdate}, 1); // Concatenate the states and actions
  //std::cout << "Size critic batch: " << input_critic_pupdate.sizes() << std::endl;
  // Use of the critic network of the current agent
  torch::Tensor value_policy = agents.at(a)->critic.forward(input_critic_pupdate);

  // Policy loss
  torch::Tensor actor_loss = -torch::mean(value_policy);

  actor_losses += actor_loss.item<float>();
  //outfile << "Actor loss: " << actor_loss << std::endl;

  agents.at(a)->optimizer_actor->zero_grad();
  actor_loss.backward();
  agents.at(a)->optimizer_actor->step();
}


/****************************************/
/****************************************/

void AACLoopFunction::PostExperiment() {
  //std::ofstream outfile;
  //outfile.open("debug_weights.txt", std::ios_base::app); // append instead of overwrite
  LOG << "Time = " << fTimeStep << std::endl;
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
  fEpisode += 1;
  //outfile << "Next episode: " << fEpisode << std::endl;
  //std::cout << "NB transitions in buffer " << buffer.size() << std::endl;
  //outfile << "Critic weights: "  << agents.at(0)->critic.parameters() << std::endl;
  //outfile << "Actor weights: "  << agents.at(0)->actor.parameters() << std::endl;
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

float AACLoopFunction::GetCriticLoss() {
  std::cout << "Critic losses: " << critic_losses << std::endl;
  float average = critic_losses / fTimeStepTraining;
  return average;
}

float AACLoopFunction::GetActorLoss() {
  float average = actor_losses / fTimeStepTraining;
  return average;
}

float AACLoopFunction::GetEntropy(){
  std::cout << "Entropies: " << entropies << std::endl;
  float average = entropies / fTimeStepTraining;
  return average;
}

REGISTER_LOOP_FUNCTIONS(AACLoopFunction, "aac_loop_functions");

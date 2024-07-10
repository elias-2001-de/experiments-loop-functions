/**
  * @file <loop-functions/example/ShelterConstrainedAccessLoopFunc.cpp>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

#include "SCALoopFunc.h"

/****************************************/
/****************************************/

SCALoopFunction::SCALoopFunction() {

}

/****************************************/
/****************************************/

SCALoopFunction::SCALoopFunction(const SCALoopFunction& orig) {}

/****************************************/
/****************************************/

SCALoopFunction::~SCALoopFunction() {}

/****************************************/

/****************************************/
void SCALoopFunction::Destroy() {}

/****************************************/
/****************************************/

void SCALoopFunction::Reset() {
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

void SCALoopFunction::Init(TConfigurationNode& t_tree) {
    // Parsing all floor circles
    TConfigurationNodeIterator it_circle("circle");
    TConfigurationNode circleParameters;
    try{
      // Finding all floor circle
      for ( it_circle = it_circle.begin( &t_tree ); it_circle != it_circle.end(); it_circle++ )
      {
          circleParameters = *it_circle;
          Circle c;
          GetNodeAttribute(circleParameters, "position", c.center);
          GetNodeAttribute(circleParameters, "radius", c.radius);
          GetNodeAttribute(circleParameters, "color", c.color);
          lCircles.push_back(c);
      }
      LOG << "number of floor circle: " << lCircles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }

    // Parsing all floor rectangles
    TConfigurationNodeIterator it_rect("rectangle");
    TConfigurationNode rectParameters;
    try{
      // Finding all floor circle
      for ( it_rect = it_rect.begin( &t_tree ); it_rect != it_rect.end(); it_rect++ )
      {
          rectParameters = *it_rect;
          Rectangle r;
          GetNodeAttribute(rectParameters, "center", r.center);
          GetNodeAttribute(rectParameters, "angle", r.angle);
          GetNodeAttribute(rectParameters, "width", r.width);
          GetNodeAttribute(rectParameters, "height", r.height);
          GetNodeAttribute(rectParameters, "color", r.color);
          lRectangles.push_back(r);
      }
      LOG << "number of floor rectangles: " << lRectangles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }
  CoreLoopFunctions::Init(t_tree);
}

/****************************************/
/****************************************/

argos::CColor SCALoopFunction::GetFloorColor(const argos::CVector2& c_position_on_plane) {
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  if (IsOnColor(vCurrentPoint, "black")) {
    return CColor::BLACK;
  }

  if (IsOnColor(vCurrentPoint, "white")) {
    return CColor::WHITE;
  }

  else {
    return CColor::GRAY50;
  }
}

/****************************************/
/****************************************/

bool SCALoopFunction::IsOnColor(CVector2& c_position_on_plane, std::string color) {
  // checking floor circles
  for (Circle c : lCircles) 
  {
    if (c.color == color)
    {
      Real d = (c.center - c_position_on_plane).Length();
      if (d <= c.radius) 
      {
        return true;
      }
    }
  }

  // checking floor rectangles
  for (Rectangle r : lRectangles) 
  {
    if (r.color == color)
    {
      Real phi = std::atan(r.height/r.width);
      Real theta = r.angle * (M_PI/180);
      Real hyp = std::sqrt((r.width*r.width) + (r.height*r.height));
      // compute position of three corner of the rectangle
      CVector2 corner1 = CVector2(r.center.GetX() - hyp*std::cos(phi + theta), r.center.GetY() + hyp*std::sin(phi + theta));
      CVector2 corner2 = CVector2(r.center.GetX() + hyp*std::cos(phi - theta), r.center.GetY() + hyp*std::sin(phi - theta));
      CVector2 corner3 = CVector2(r.center.GetX() + hyp*std::cos(phi + theta), r.center.GetY() - hyp*std::sin(phi + theta));
      // computing the three vectors
      CVector2 corner2ToCorner1 = corner1 - corner2; 
      CVector2 corner2ToCorner3 = corner3 - corner2; 
      CVector2 corner2ToPos = c_position_on_plane - corner2; 
      // compute the four inner products
      Real ip1 = corner2ToPos.GetX()*corner2ToCorner1.GetX() + corner2ToPos.GetY()*corner2ToCorner1.GetY();
      Real ip2 = corner2ToCorner1.GetX()*corner2ToCorner1.GetX() + corner2ToCorner1.GetY()*corner2ToCorner1.GetY();
      Real ip3 = corner2ToPos.GetX()*corner2ToCorner3.GetX() + corner2ToPos.GetY()*corner2ToCorner3.GetY();
      Real ip4 = corner2ToCorner3.GetX()*corner2ToCorner3.GetX() + corner2ToCorner3.GetY()*corner2ToCorner3.GetY();
      if (ip1 > 0 && ip1 < ip2 && ip3 > 0 && ip3 < ip4)
      {
        return true;
      }
    }
  }
  return false;
}

/****************************************/
/****************************************/

void SCALoopFunction::PreStep() {
  int N = (critic_input_dim - 4)/5; // Number of closest neighbors to consider (4 absolute states + 5 relative states)
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
    if (IsOnColor(cEpuckPosition, "white")) {
      reward = 1.0;
    } else {
      reward = 0;//std::pow(10, -3 * fDistanceSpot / m_fDistributionRadius);
    }
    rewards.push_back(torch::tensor(reward));
    m_fObjectiveFunction += reward;
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
        i++;
      } catch (std::exception &ex) {
        LOGERR << "Error while setting network: " << ex.what() << std::endl;
      }
    }

    fTimeStep += 1;
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

void SCALoopFunction::PostStep() {}

/****************************************/
/****************************************/

void SCALoopFunction::PostExperiment() {
  LOG << "Time = " << fTimeStep << std::endl;
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
}

/****************************************/
/****************************************/

Real SCALoopFunction::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

float SCALoopFunction::GetTDError() {
  float sum = std::accumulate(TDerrors.begin(), TDerrors.end(), 0.0f);
  float average = sum / TDerrors.size();
  return average;
}

/****************************************/
/****************************************/

float SCALoopFunction::GetEntropy() {
  float sum = std::accumulate(Entropies.begin(), Entropies.end(), 0.0f);
  float average = sum / Entropies.size();
  return average;
}

/****************************************/
/****************************************/

float SCALoopFunction::GetActorLoss() {
  float sum = std::accumulate(actor_losses.begin(), actor_losses.end(), 0.0f);
  float average = sum / actor_losses.size();
  return average;
}


/****************************************/
/****************************************/

float SCALoopFunction::GetCriticLoss() {
  float sum = std::accumulate(critic_losses.begin(), critic_losses.end(), 0.0f);
  float average = sum / critic_losses.size();
  return average;
}

/****************************************/
/****************************************/

std::vector<float> SCALoopFunction::GetBehavHist() {
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

std::vector<RelativePosition> SCALoopFunction::compute_relative_positions(const CVector2& base_position, const CRadians& base_yaw, const std::vector<std::pair<CVector2, CRadians>>& all_positions) {
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

CVector3 SCALoopFunction::GetRandomPosition() {
  Real temp;
  Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
  Real b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));

  Real fPosX = (a * 2*0.5) - 0.5;
  Real fPosY = (b * 2*1.5) - 1.5;

  return CVector3(fPosX, fPosY, 0);
}

/****************************************/
/****************************************/

void SCALoopFunction::SetTraining(bool value) {
    training = value;
}

REGISTER_LOOP_FUNCTIONS(SCALoopFunction, "sca_loop_functions");

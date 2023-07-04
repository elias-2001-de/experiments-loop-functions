/**
  * @file <loop-functions/AggregationTwoSpotsLoopFunc.cpp>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @license MIT License
  */

#include "AggregationTwoSpotsXOR2.h"

/****************************************/
/****************************************/

AggregationTwoSpotsXOR2::AggregationTwoSpotsXOR2() {
  m_fRadius = 0.3;
  m_cCoordSpot1 = CVector2(0.5,0);
  m_cCoordSpot2 = CVector2(-0.5,0);
  m_unScoreSpot1 = 0;
  m_unScoreSpot2 = 0;
  m_fObjectiveFunction = 0;
}

/****************************************/
/****************************************/

AggregationTwoSpotsXOR2::AggregationTwoSpotsXOR2(const AggregationTwoSpotsXOR2& orig) {}

/****************************************/
/****************************************/

void AggregationTwoSpotsXOR2::Init(TConfigurationNode& t_tree) {
    CoreLoopFunctions::Init(t_tree);

    m_context = zmq::context_t(1);
    m_socket_actor = zmq::socket_t(m_context, ZMQ_PUSH);
    m_socket_actor.connect("tcp://localhost:5555");

    m_socket_critic = zmq::socket_t(m_context, ZMQ_PUSH);
    m_socket_critic.connect("tcp://localhost:5556");
}

/****************************************/
/****************************************/


AggregationTwoSpotsXOR2::~AggregationTwoSpotsXOR2() {}

/****************************************/
/****************************************/

void AggregationTwoSpotsXOR2::Destroy() {}

/****************************************/
/****************************************/

argos::CColor AggregationTwoSpotsXOR2::GetFloorColor(const argos::CVector2& c_position_on_plane) {
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  Real d = (m_cCoordSpot1 - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::BLACK;
  }

  d = (m_cCoordSpot2 - vCurrentPoint).Length();
  if (d <= m_fRadius) {
    return CColor::BLACK;
  }

  return CColor::GRAY50;
}


/****************************************/
/****************************************/

void AggregationTwoSpotsXOR2::Reset() {
  m_fObjectiveFunction = 0;
  m_unScoreSpot1 = 0;
  m_unScoreSpot2 = 0;
  CoreLoopFunctions::Reset();
}

/****************************************/
/****************************************/

void AggregationTwoSpotsXOR2::PostStep() {
    m_unScoreSpot1 = 0;
    m_unScoreSpot2 = 0;
    CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0,0);
    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
        CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
        cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                           pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
        Real fDistanceSpot1 = (m_cCoordSpot1 - cEpuckPosition).Length();
        Real fDistanceSpot2 = (m_cCoordSpot2 - cEpuckPosition).Length();
        if (fDistanceSpot1 <= m_fRadius) {
         m_unScoreSpot1 += 1;
        } else if (fDistanceSpot2 <= m_fRadius){
         m_unScoreSpot2 += 1;
        }
    }
    //LOG << "Sco1 = " << m_unScoreSpot1 << "Sco2 = " << m_unScoreSpot2 << std::endl;
    m_fObjectiveFunction += Max(m_unScoreSpot1, m_unScoreSpot2);

    // send message to policy manager
    Data data {{1, 2, 3}, {4, 5, 6}};
    std::string serialized_data = serialize(data);

    zmq::message_t message(serialized_data.size());
    memcpy(message.data(), serialized_data.c_str(), serialized_data.size());

    m_socket_actor.send(message, zmq::send_flags::dontwait);

    message.rebuild(serialized_data.size());
    memcpy(message.data(), serialized_data.c_str(), serialized_data.size());
    m_socket_critic.send(message, zmq::send_flags::dontwait);

}


void AggregationTwoSpotsXOR2::PostExperiment() {
  LOG << "Score = " << m_fObjectiveFunction << std::endl;
}

/****************************************/
/****************************************/

Real AggregationTwoSpotsXOR2::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

CVector3 AggregationTwoSpotsXOR2::GetRandomPosition() {
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

REGISTER_LOOP_FUNCTIONS(AggregationTwoSpotsXOR2, "aggregation_xor2");

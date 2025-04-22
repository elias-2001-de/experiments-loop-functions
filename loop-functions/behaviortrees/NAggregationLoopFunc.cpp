#include "NAggregationLoopFunc.h"

using namespace argos;

const int NAggregationLoopFunction::_area_count;
const float NAggregationLoopFunction::_area_to_origin_distance = 0.7;
const float NAggregationLoopFunction::_area_radius = 0.25;

NAggregationLoopFunction::NAggregationLoopFunction() :
    _area_centers(),
    _objective_function(0.0f)
{
    // setup areas
    for(int i = 0; i < _area_count; ++i)
    {
        float angle = 2.0f * M_PI / static_cast<float>(_area_count)
                                  * static_cast<float>(i);
        _area_centers[i] = CVector2(
            _area_to_origin_distance * std::cos(angle),
            _area_to_origin_distance * std::sin(angle));

        std::cout << _area_centers[i].GetX() << ","
                  << _area_centers[i].GetY() << std::endl;
    }

    //Reset();
}

NAggregationLoopFunction::~NAggregationLoopFunction()
{}

void NAggregationLoopFunction::Destroy()
{}

void NAggregationLoopFunction::Init(TConfigurationNode& t_tree)
{
    CoreLoopFunctions::Init(t_tree);
}

argos::CColor NAggregationLoopFunction::GetFloorColor(
    const argos::CVector2& c_position_on_plane)
{
    return (getAreaIndex(c_position_on_plane) >= 0) ?
        CColor::BLACK :
        CColor::GRAY50;
}

void NAggregationLoopFunction::PostStep()
{
    CSpace::TMapPerType& epuck_map = GetSpace().GetEntitiesByType("epuck");
    int epuck_count = epuck_map.size();
    int epuck_per_area = epuck_count / _area_count;

    int epuck_count_per_area[_area_count] = {0};

    // count the number of robots in each area
    CSpace::TMapPerType::iterator it = epuck_map.begin();
    for(; it != epuck_map.end(); ++it)
    {
        CEPuckEntity* epuck = any_cast<CEPuckEntity*>(it->second);
        CVector3 epuck_pos3 = epuck->GetEmbodiedEntity().GetOriginAnchor().Position;
        CVector2 epuck_pos(epuck_pos3.GetX(), epuck_pos3.GetY());

        // test if robot is in area
        int area_index = getAreaIndex(epuck_pos);
        if(area_index >= 0)
        {
            ++epuck_count_per_area[area_index];
        }
    }

    // compute step score
    Real step_score = 0.0f;
    for(int i = 0; i < _area_count; ++i)
    {
        Real diff = std::abs(epuck_count_per_area[i] - epuck_per_area);
        Real area_score = epuck_per_area - diff;
        step_score += std::max(Real(0.0f), area_score);
    }
    LOG << "step score: " << step_score << std::endl;

    // update global score
    _objective_function += step_score;
}

void NAggregationLoopFunction::Reset()
{
    CoreLoopFunctions::Reset();
    _objective_function = 0.0f;
}

CVector3 NAggregationLoopFunction::GetRandomPosition()
{
    Real temp;
    Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    Real b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    // If b < a, swap them
    if (b < a) {
    temp = a;
    a = b;
    b = temp;
    }
    Real fPosX = b * m_fDistributionRadius *
        cos(2.0f * CRadians::PI.GetValue() * (a/b));
    Real fPosY = b * m_fDistributionRadius *
        sin(2.0f * CRadians::PI.GetValue() * (a/b));

    return CVector3(fPosX, fPosY, 0);
}

int NAggregationLoopFunction::getAreaIndex(
    argos::CVector2 const& position) const
{
    // test areas
    for(int i = 0; i < _area_count; ++i)
    {
        float dist_to_area_center = (position - _area_centers[i]).Length();

        if(dist_to_area_center <= _area_radius)
        {
            return i;
        }
    }

    // else
    return -1;
}

REGISTER_LOOP_FUNCTIONS(NAggregationLoopFunction,
                        "n-aggregation_loop_function");


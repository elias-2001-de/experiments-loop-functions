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

  CoreLoopFunctions::Reset();
}

/****************************************/
/****************************************/

void Homing::PostStep() {
  at::Tensor grid = torch::zeros({50, 50});
  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  m_unScoreSpot1 = 0;	
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
    
    // Scale the position to the grid size
    int grid_x = static_cast<int>(std::round((cEpuckPosition.GetX() + 1.231) / 2.462 * 49));
    int grid_y = static_cast<int>(std::round((-cEpuckPosition.GetY() + 1.231) / 2.462 * 49));
    
    Real fDistanceSpot1 = (m_cCoordSpot1 - cEpuckPosition).Length();
    if (fDistanceSpot1 <= m_fRadius) {
      m_unScoreSpot1 += 1;    // Reward
      m_fObjectiveFunction += 1;
      //std::cout << "Robot at [" << grid_x << ";" << grid_y << "]\n";      
    }
    
    // Set the grid cell that corresponds to the epuck's position to 1
    grid[grid_x][grid_y] = 1;
  }
  
  //if(m_unScoreSpot1 > 0){
  //std::cout << "score: " << m_unScoreSpot1 << std::endl;	  
  // place spot in grid
  int grid_x = static_cast<int>(std::round((m_cCoordSpot1.GetX() + 1.231) / 2.462 * 49));
  int grid_y = static_cast<int>(std::round((-m_cCoordSpot1.GetY() + 1.231) / 2.462 * 49));
  int radius = static_cast<int>(std::round(0.35 / 2.462 * 49));
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
}

/****************************************/
/****************************************/

void Homing::PostExperiment() {
  //LOG << "Score = " << m_fObjectiveFunction << std::endl;
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

void Homing::print_grid(at::Tensor grid){
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

REGISTER_LOOP_FUNCTIONS(Homing, "homing_test");

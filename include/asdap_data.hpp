#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h" // TODO: check geometry central includes
#include <igl/min_quad_with_fixed.h>

#include "asdap_utils.hpp"

namespace gc = geometrycentral;
namespace gcs = gc::surface;

enum class BENDING {
    DIHEDRAL = 0,
    REFERENCE = 1,
    SVD = 2,
    NONE = 3
};
// std::vector<std::string> bmode_name = {"DIHEDRAL", "REFERENCE", "SVD"};

struct ASDAPData {
  // input (original vertices & faces)
  Eigen::Matrix<double, -1, 3> V;
  Eigen::Matrix<int, -1, 3> F;
  std::vector<Eigen::Matrix3d> O_inv; // face orientation matrix inverses
  BENDING bmode;

  std::vector<bool> changedV;
  std::vector<bool> nextChangedV;
  Eigen::MatrixXd last_gradients;

  bool hasConverged;
  int iteration;

  // the minimal size for a gradient during optimisation (All gradients smaller than it will be truncated to 0) // TODO: implement initialisation in asdap_precompute function
  double minimumGradient;
  
  // geometries
  std::unique_ptr<gcs::ManifoldSurfaceMesh> inputMesh;
  std::unique_ptr<gcs::VertexPositionGeometry> inputGeometry;
};

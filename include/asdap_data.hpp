#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h" // TODO: check geometry central includes
#include <igl/min_quad_with_fixed.h>

namespace gc = geometrycentral;
namespace gcs = gc::surface;

struct ASDAPData {
  // input (original vertices & faces)
  Eigen::Matrix<double, -1, 3> V;
  Eigen::Matrix<int, -1, 3> F;

  std::vector<bool> changedV;
  std::vector<bool> nextChangedV;

  bool hasConverged;
  int iteration;
  
  // geometries
  std::unique_ptr<gcs::ManifoldSurfaceMesh> inputMesh;
  std::unique_ptr<gcs::VertexPositionGeometry> inputGeometry;
};

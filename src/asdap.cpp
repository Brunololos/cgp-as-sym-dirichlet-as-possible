// TODO: implement asdap.hpp
#include "asdap.hpp"

#include <iostream>
#include <fstream>

#include <igl/parallel_for.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/polar_svd3x3.h>

#include <Eigen/Sparse>

// TODO: implement segments.hpp
#include "segments.hpp"

typedef Eigen::SparseVector<double>::InnerIterator SVIter;


void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data){
  data.V = V;
  data.F = F;
  data.inputMesh.reset(new gcs::ManifoldSurfaceMesh(F));
  data.inputGeometry.reset(new gcs::VertexPositionGeometry(*data.inputMesh, V));
  data.intTri.reset(new gcs::SignpostIntrinsicTriangulation(*data.inputMesh, *data.inputGeometry));
  data.intTri->flipToDelaunay();
  data.intTri->requireEdgeCotanWeights();
  data.intTri->requireEdgeLengths();
}

void build_intrinsic_mesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data){
  initMesh(/*In:*/V, F, /*Out:*/data);

  calc_segments(/*In:*/data.V, data.intTri, /*Out:*/data.all_segments);
  calc_segments_per_vertex(/*In:*/data.V, data.all_segments, /*Out:*/data.K, data.L, data.Bs, data.segments, data.weights);
}

void asdap_precompute(const Eigen::VectorXi& constraints_indices, ASDAPData& data){
  // TODO: implement asdap_precompute (i.e. calculation of initial solution)
}

void local_step(const Eigen::MatrixXd& U, ASDAPData& data){
  // TODO: implement local step for asdap
}

void global_step(const Eigen::MatrixXd& constraints, const ASDAPData& data, Eigen::MatrixXd& U){
  // TODO: implement global step for asdap
}

void asdap_solve(const Eigen::MatrixXd& constraints, ASDAPData& data, Eigen::MatrixXd& U){
  // TODO: implement asdap_solve
}

double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U){
  double result = 0;
  // TODO: implement asdap_energy
  return result;
}

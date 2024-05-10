#pragma once

// TODO: implement asdap_data.hpp
#include "asdap_data.hpp"

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data);
void build_intrinsic_mesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data);
void asdap_precompute(const Eigen::VectorXi& constraints_indices, ASDAPData& data);
void asdap_solve(const Eigen::MatrixXd& constraints, ASDAPData& data, Eigen::MatrixXd& U);
double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U);

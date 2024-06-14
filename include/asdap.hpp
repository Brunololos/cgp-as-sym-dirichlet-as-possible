#pragma once

#include "asdap_data.hpp"

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data);

void asdap_precompute(const Eigen::VectorXi& constraints_indices, const Eigen::MatrixXd& constraints, ASDAPData& data);

std::pair<Eigen::MatrixXd, double> asdap_step(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U);

std::pair<Eigen::MatrixXd, double> asdap_optim(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U);

double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U, const ENERGY type);

std::pair<Eigen::MatrixXd, double> asdap_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U);

std::pair<Eigen::Matrix3d, double> asdap_energy_face_vertex_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<size_t>& face);
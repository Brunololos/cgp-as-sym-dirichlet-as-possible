#pragma once

#include "asdap_data.hpp"
#include "asdap_utils.hpp"

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data);
void asdap_set_constraints(const Eigen::VectorXi& constraints_indices, const Eigen::MatrixXd& constraints, ASDAPData& data);

std::pair<Eigen::MatrixXd, double> asdap_step(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U);
std::pair<Eigen::MatrixXd, double> asdap_optim(const Eigen::VectorXi& constraints_indices, double lr, int max_iterations, ASDAPData& data, Eigen::MatrixXd& U);

double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U);
std::pair<double, double> asdap_energies(const ASDAPData& data, const Eigen::MatrixXd& U);
double asdap_scaling_energy(const ASDAPData& data, const Eigen::MatrixXd& U, const ENERGY type);
double asdap_local_rotation_energy(const ASDAPData& data, const Eigen::MatrixXd& U);

std::pair<Eigen::MatrixXd, double> asdap_facewise_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U);
std::pair<Eigen::Matrix3d, double> asdap_energy_face_scaling_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face);
std::pair<Eigen::Matrix3d, double> asdap_energy_local_svd_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx);
std::pair<Eigen::Matrix3d, double> asdap_energy_non_rigid_local_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx);
std::pair<Eigen::Matrix3d, double> asdap_energy_dihedral_local_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx);
std::pair<Eigen::Matrix3d, double> asdap_energy_global_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx, const Eigen::Matrix3d avg_rotation);
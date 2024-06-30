#pragma once

#include "asdap_data.hpp"
#include "asdap_utils.hpp"

std::pair<Eigen::MatrixXd, double> asdap_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U);
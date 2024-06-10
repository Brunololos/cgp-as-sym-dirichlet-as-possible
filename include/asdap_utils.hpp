#ifndef ASDAP_UTILS_H
#define ASDAP_UTILS_H

#include <TinyAD/Scalar.hh>
#include "asdap_data.hpp" // TODO: possibly invert this dependence at some point, such that asdap_data depends on asdap_utils

using ADFloat = TinyAD::Float<3>;
using ADDouble = TinyAD::Double<3>;
using ADDouble9 = TinyAD::Double<9>;

void printEigenMatrix(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M);
void printTinyADMatrix(std::string name, Eigen::Matrix<ADDouble, Eigen::Dynamic, Eigen::Dynamic> M);

#endif
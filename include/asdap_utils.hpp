#ifndef ASDAP_UTILS_H
#define ASDAP_UTILS_H

#include <TinyAD/Scalar.hh>
//#include "asdap_data.hpp" // TODO: possibly invert this dependence at some point, such that asdap_data depends on asdap_utils

using ADFloat = TinyAD::Float<3>;
using ADDouble = TinyAD::Double<3>;
using ADDouble9 = TinyAD::Double<9>;

enum class ENERGY {
    DETERMINANT,
    SUMOFSQUARES
};

void printEigenMatrixXi(std::string name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> M);
void printEigenMatrixXd(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M);
void printTinyADMatrix(std::string name, Eigen::Matrix<ADDouble, Eigen::Dynamic, Eigen::Dynamic> M);
void printTinyAD9Matrix(std::string name, Eigen::Matrix<ADDouble9, Eigen::Dynamic, Eigen::Dynamic> M);

Eigen::Matrix3d calcTriangleOrientation(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector3d v2);
Eigen::Matrix3<ADDouble9> calcTinyAD9TriangleOrientation(Eigen::Vector3<ADDouble9> v0, Eigen::Vector3<ADDouble9> v1, Eigen::Vector3<ADDouble9> v2);

#endif
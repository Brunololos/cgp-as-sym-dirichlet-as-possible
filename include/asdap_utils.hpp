#ifndef ASDAP_UTILS_H
#define ASDAP_UTILS_H

#define _USE_MATH_DEFINES

#include <iomanip>
#include <math.h>
#include "tinyad_defs.hpp"
#include "my_svd.hpp"
#include "print.hpp"

enum class ENERGY {
    DETERMINANT,
    SUMOFSQUARES
};

enum class RAXIS {
    YAW = 0,
    PITCH = 1,
    ROLL = 2,
    NONE = 3
};

Eigen::MatrixXd rotate_selected(Eigen::MatrixXd U, Eigen::VectorXi selected_ids, RAXIS rot_axis, double angle);

Eigen::Matrix3d calcTriangleOrientation(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector3d v2, bool calcRightOrientation = true);
Eigen::Matrix3<ADDouble9> calcTinyAD9TriangleOrientation(Eigen::Vector3<ADDouble9> v0, Eigen::Vector3<ADDouble9> v1, Eigen::Vector3<ADDouble9> v2, bool calcRightOrientation = true);

Eigen::Matrix3d calcTriangleRotation(Eigen::Matrix3d p, Eigen::Matrix3d q);
Eigen::Matrix3<ADDouble9> calcTriangleRotation(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3d q);
Eigen::Matrix3<ADDouble9> calcTriangleRotation(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3<ADDouble9> q);

double calcTriangleDihedralAngle(Eigen::Matrix3d p, Eigen::Matrix3d q);
ADDouble9 calcTriangleDihedralAngle(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3d q);

double calcEigenSquaredFrobenius(Eigen::Matrix3d matrix);
ADDouble9 calcTinyAD9SquaredFrobenius(Eigen::Matrix3<ADDouble9> matrix);

double calcRotationAngle(Eigen::Matrix3d P);
ADDouble9 calcRotationAngle(Eigen::Matrix3<ADDouble9> P);
Eigen::Vector3d calcRotationAxis(Eigen::Matrix3d P);
Eigen::Vector3<ADDouble9> calcRotationAxis(Eigen::Matrix3<ADDouble9> P);

double calcRotationDifference(Eigen::Matrix3d P, Eigen::Matrix3d Q);
ADDouble9 calcRotationDifference(Eigen::Matrix3d P, Eigen::Matrix3<ADDouble9> Q);
ADDouble9 calcRotationDifference(Eigen::Matrix3<ADDouble9> P, Eigen::Matrix3d Q);
ADDouble9 calcRotationDifference(Eigen::Matrix3<ADDouble9> P, Eigen::Matrix3<ADDouble9> Q);

#endif
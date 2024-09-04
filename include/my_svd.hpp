#ifndef MYSVD_H
#define MYSVD_H

#include "tinyad_defs.hpp"
#include "print.hpp"

std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>> my_svd(Eigen::MatrixX<ADDouble9> M);
std::pair<Eigen::VectorX<ADDouble9>, Eigen::MatrixX<ADDouble9>> my_evd(Eigen::MatrixX<ADDouble9> M, Eigen::VectorX<ADDouble9> eigenvalues = Eigen::VectorX<ADDouble9>::Zero(0), Eigen::MatrixX<ADDouble9> eigenvectors = Eigen::MatrixX<ADDouble9>::Zero(0, 0), int eig_count = 0);
std::pair<Eigen::MatrixX<ADDouble9>, Eigen::VectorX<ADDouble9>> my_jordan_gaussian_transform(Eigen::MatrixX<ADDouble9> M);
Eigen::MatrixX<ADDouble9> calc_my_hermitian(Eigen::VectorX<ADDouble9> eigenvector);
Eigen::MatrixX<ADDouble9> calc_my_hermitian_inverse(Eigen::VectorX<ADDouble9> eigenvector);
Eigen::MatrixX<ADDouble9> get_my_reduced_matrix(Eigen::MatrixX<ADDouble9> M, int new_size);
Eigen::MatrixX<ADDouble9> diagonal_inverse(Eigen::MatrixX<ADDouble9> M);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> my_svdd(Eigen::MatrixXd M);
std::pair<Eigen::VectorXd, Eigen::MatrixXd> my_evdd(Eigen::MatrixXd M, Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(0), Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Zero(0, 0), int eig_count = 0);
std::pair<Eigen::MatrixXd, Eigen::VectorXd> my_jordan_gaussian_transformd(Eigen::MatrixXd M);
Eigen::MatrixXd calc_my_hermitiand(Eigen::VectorXd eigenvector);
Eigen::MatrixXd calc_my_hermitian_inversed(Eigen::VectorXd eigenvector);
Eigen::MatrixXd get_my_reduced_matrixd(Eigen::MatrixXd M, int new_size);
Eigen::MatrixXd diagonal_inversed(Eigen::MatrixXd M);

#endif
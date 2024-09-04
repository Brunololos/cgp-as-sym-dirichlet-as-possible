#ifndef PRINT_H
#define PRINT_H

#include "tinyad_defs.hpp"

// Reset
static const std::string RESET = "\033[0m";

// Regular Colors
static const std::string BLACK = "\033[0;30m";
static const std::string RED = "\033[0;31m";
static const std::string GREEN = "\033[0;32m";
static const std::string YELLOW = "\033[0;33m";
static const std::string BLUE = "\033[0;34m";
static const std::string PURPLE = "\033[0;35m";
static const std::string CYAN = "\033[0;36m";
static const std::string WHITE = "\033[0;37m";
static const std::string GRAY = "\033[0;90m";

void printEigenMatrixXi(std::string name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> M);
void printEigenMatrixXd(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M);
void printEigenMatrixXcd(std::string name, Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> M);
void printTinyADMatrix(std::string name, Eigen::Matrix<ADDouble, Eigen::Dynamic, Eigen::Dynamic> M);
void printTinyAD9Matrix(std::string name, Eigen::Matrix<ADDouble9, Eigen::Dynamic, Eigen::Dynamic> M);

std::string repeat(std::string string, int number_of_repetitions);
std::string dye(std::string string, std::string color);

#endif
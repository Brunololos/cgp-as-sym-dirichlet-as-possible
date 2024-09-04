#include "print.hpp"

void printEigenMatrixXi(std::string name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> M)
{
    int n_zero_grads = 0;
    bool found_zero_grad = true;
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
        found_zero_grad = true;
        for (int j=0; j<M.cols(); j++)
        {
            if (M(i, j) != 0)
            {
                found_zero_grad = false;
                break;
            }
        }
        if (found_zero_grad) { n_zero_grads++; continue; }
        if (n_zero_grads > 0) {
            for (int j=0; j<M.cols(); j++)
            {
                std::cout << "[ 0 ]";
            }
            std::cout << " x " << n_zero_grads << " times" << std::endl;
            n_zero_grads = 0;
        }
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ " << M(i, j) << " ]";
        }
        std::cout << std::endl;
    }

    if (n_zero_grads > 0) {
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ 0 ]";
        }
        std::cout << " x " << n_zero_grads << " times" << std::endl;
        n_zero_grads = 0;
    }
    std::cout << std::endl;
}

void printEigenMatrixXd(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M)
{
    int n_zero_grads = 0;
    bool found_zero_grad = true;
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
        found_zero_grad = true;
        for (int j=0; j<M.cols(); j++)
        {
            if (M(i, j) != 0)
            {
                found_zero_grad = false;
                break;
            }
        }
        if (found_zero_grad) { n_zero_grads++; continue; }
        if (n_zero_grads > 0) {
            for (int j=0; j<M.cols(); j++)
            {
                std::cout << "[ 0 ]";
            }
            std::cout << " x " << n_zero_grads << " times" << std::endl;
            n_zero_grads = 0;
        }
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ " << M(i, j) << " ]";
        }
        std::cout << std::endl;
    }

    if (n_zero_grads > 0) {
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ 0 ]";
        }
        std::cout << " x " << n_zero_grads << " times" << std::endl;
        n_zero_grads = 0;
    }
    std::cout << std::endl;
}

void printEigenMatrixXcd(std::string name, Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> M)
{
    int n_zero_grads = 0;
    bool found_zero_grad = true;
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
        found_zero_grad = true;
        for (int j=0; j<M.cols(); j++)
        {
            if (M(i, j) != (std::complex<double>) 0)
            {
                found_zero_grad = false;
                break;
            }
        }
        if (found_zero_grad) { n_zero_grads++; continue; }
        if (n_zero_grads > 0) {
            for (int j=0; j<M.cols(); j++)
            {
                std::cout << "[ 0 ]";
            }
            std::cout << " x " << n_zero_grads << " times" << std::endl;
            n_zero_grads = 0;
        }
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ " << M(i, j) << " ]";
        }
        std::cout << std::endl;
    }

    if (n_zero_grads > 0) {
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ 0 ]";
        }
        std::cout << " x " << n_zero_grads << " times" << std::endl;
        n_zero_grads = 0;
    }
    std::cout << std::endl;
}

void printTinyADMatrix(std::string name, Eigen::Matrix<ADDouble, Eigen::Dynamic, Eigen::Dynamic> M)
{
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ " << M(i, j).val << " ]";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printTinyAD9Matrix(std::string name, Eigen::Matrix<ADDouble9, Eigen::Dynamic, Eigen::Dynamic> M)
{
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
        for (int j=0; j<M.cols(); j++)
        {
            std::cout << "[ " << M(i, j).val << " ]";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::string repeat(std::string string, int number_of_repetitions)
{
  std::string string_tiling = "";
  for(int i = 0; i < number_of_repetitions; i++)
  {
    string_tiling += string;
  }
  return string_tiling;
}

std::string dye(std::string string, std::string color)
{
  // TODO: Check if color is valid
  return color + string + RESET;
}
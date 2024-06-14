#include "asdap_utils.hpp"

void printEigenMatrix(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M)
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
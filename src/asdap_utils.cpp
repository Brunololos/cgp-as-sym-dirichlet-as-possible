#include "asdap_utils.hpp"

void printEigenMatrix(std::string name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M)
{
    std::cout << name  << " matrix:" << std::endl;
    for (int i=0; i<M.rows(); i++)
    {
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
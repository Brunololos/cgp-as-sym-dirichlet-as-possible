#include "asdap_utils.hpp"

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

Eigen::Matrix3d calcTriangleOrientation(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector3d v2)
{
    Eigen::Vector3d lead({1, 0, 0});
    Eigen::Vector3d center({0, 1, 0});

    // calculate local triangle from input triangle
    Eigen::Vector3d localv1 = v1 - v0;
    Eigen::Vector3d localv2 = v2 - v0;

    // calculate canonical reference triangle
    // => triangle lies in the xy-plane
    // => triangle vertex v0 lies in the origin (local coordinate system)
    // => triangle vertex v1 lies on the x-axis (edge (v0, v1) coincides with the x-axis)
    // => triangle vertex v2 lies within the first quadrant of the local coordinate system (this sometimes flips triangles, because of the abs() TODO: check if this should be the case or if it does not matter)
    Eigen::Vector3d localv1v2normal = localv1.cross(localv2).normalized();
    Eigen::Vector3d localv1v1v2normal = localv1.cross(localv1v2normal).normalized();
    double c2y = abs(localv2.dot(localv1v1v2normal));

    Eigen::Vector3d canonv1 = localv1.norm()*lead;
    Eigen::Vector3d canonv2 = (localv1.dot(localv2)/localv1.norm())*lead + c2y*center;

    // Calculate rotation matrix that transforms the canonical triangle into the local one
    double R00 = localv1(0) / canonv1(0);
    double R10 = localv1(1) / canonv1(0);
    double R20 = localv1(2) / canonv1(0);

    double R01 = (localv2(0) - canonv2(0)*R00) / canonv2(1);
    double R11 = (localv2(1) - canonv2(0)*R10) / canonv2(1);
    double R21 = (localv2(2) - canonv2(0)*R20) / canonv2(1);

    double R02 = localv1v2normal(0);
    double R12 = localv1v2normal(1);
    double R22 = localv1v2normal(2);

    Eigen::MatrixXd R(3, 3);
    R << R00, R01, R02,
         R10, R11, R12,
         R20, R21, R22;

/*     Eigen::MatrixXd R_inv = R.inverse();

    Eigen::MatrixXd IT(v0.rows(), v0.cols() + v1.cols() + v2.cols());
    IT << v0, v1, v2;
    printEigenMatrixXd("INPUT TRIANGLE", IT);

    Eigen::MatrixXd LT(localv1.rows(), localv1.cols() + localv1.cols() + localv2.cols());
    LT << Eigen::Vector3d({0, 0, 0}), localv1, localv2;
    printEigenMatrixXd("Localised TRIANGLE", LT);

    std::cout << "localv1v2 normal: (" << localv1v2normal(0) << ", " << localv1v2normal(1) << ", " << localv1v2normal(2) << ")" << std::endl;
    std::cout << "localv1v1v2 normal: (" << localv1v1v2normal(0) << ", " << localv1v1v2normal(1) << ", " << localv1v1v2normal(2) << ")" << std::endl << std::endl;

    Eigen::MatrixXd CT(canonv1.rows(), canonv1.cols() + canonv1.cols() + canonv2.cols());
    CT << Eigen::Vector3d({0, 0, 0}), canonv1, canonv2;
    printEigenMatrixXd("Canonical TRIANGLE", CT);

    printEigenMatrixXd("Rotation", R);
    printEigenMatrixXd("Inverse Rotation", R_inv);

    Eigen::MatrixXd RCT(3, 3);
    RCT << R_inv * Eigen::Vector3d({0, 0, 0}), R_inv * localv1, R_inv * localv2;
    printEigenMatrixXd("Canonical Triangle Reconstruction", RCT);

    Eigen::Vector3d fn = R * Eigen::Vector3d({0, 0, 1});
    fn.normalize();
    std::cout << "forward calced normal: (" << fn(0) << ", " << fn(1) << ", " << fn(2) << ")" << std::endl;

    Eigen::Vector3d rn = R_inv * localv1.cross(localv2).normalized();
    rn.normalize();
    std::cout << "reconstructed normal: (" << rn(0) << ", " << rn(1) << ", " << rn(2) << ")" << std::endl; */

    return R;
}

Eigen::Matrix3<ADDouble9> calcTinyAD9TriangleOrientation(Eigen::Vector3<ADDouble9> v0, Eigen::Vector3<ADDouble9> v1, Eigen::Vector3<ADDouble9> v2)
{
    Eigen::Vector3d lead({1, 0, 0});
    Eigen::Vector3d center({0, 1, 0});

    // calculate local triangle from input triangle
    Eigen::Vector3<ADDouble9> localv1 = v1 - v0;
    Eigen::Vector3<ADDouble9> localv2 = v2 - v0;

    // calculate canonical reference triangle
    // => triangle lies in the xy-plane
    // => triangle vertex v0 lies in the origin (local coordinate system)
    // => triangle vertex v1 lies on the x-axis (edge (v0, v1) coincides with the x-axis)
    // => triangle vertex v2 lies within the first quadrant of the local coordinate system (this sometimes flips triangles, because of the abs() TODO: check if this should be the case or if it does not matter)
    Eigen::Vector3<ADDouble9> localv1v2normal = localv1.cross(localv2).normalized();
    Eigen::Vector3<ADDouble9> localv1v1v2normal = localv1.cross(localv1v2normal).normalized();
    ADDouble9 c2y = abs(localv2.dot(localv1v1v2normal)); //c2y = abs(c2y);

    Eigen::Vector3<ADDouble9> canonv1 = localv1.norm()*lead;
    Eigen::Vector3<ADDouble9> canonv2 = (localv1.dot(localv2)/localv1.norm())*lead + c2y*center;

    // Calculate rotation matrix that transforms the canonical triangle into the local one
    ADDouble9 R00 = localv1(0) / canonv1(0);
    ADDouble9 R10 = localv1(1) / canonv1(0);
    ADDouble9 R20 = localv1(2) / canonv1(0);

    ADDouble9 R01 = (localv2(0) - canonv2(0)*R00) / canonv2(1);
    ADDouble9 R11 = (localv2(1) - canonv2(0)*R10) / canonv2(1);
    ADDouble9 R21 = (localv2(2) - canonv2(0)*R20) / canonv2(1);

    ADDouble9 R02 = localv1v2normal(0);
    ADDouble9 R12 = localv1v2normal(1);
    ADDouble9 R22 = localv1v2normal(2);

    Eigen::MatrixX<ADDouble9> R(3, 3);
    R << R00, R01, R02,
         R10, R11, R12,
         R20, R21, R22;
    return R;
}
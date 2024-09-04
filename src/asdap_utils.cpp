#include "asdap_utils.hpp"

Eigen::MatrixXd rotate_selected(Eigen::MatrixXd U, Eigen::VectorXi selected_ids, RAXIS rot_axis, double angle)
{
  angle = (0.5 / 360.0) * 2.0 * 3.14159265;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (int i=0; i<selected_ids.size(); i++)
  {
    double id = selected_ids(i);
    mean += U.row(id);
  }
  mean = mean / selected_ids.size();

  double u0, u1, u2, p0, p1, p2;
  for (int i=0; i<selected_ids.size(); i++)
  {
    double id = selected_ids(i);
    u0 = U(id, 0) - mean(0);
    u1 = U(id, 1) - mean(1);
    u2 = U(id, 2) - mean(2);
    switch (rot_axis)
    {
      case RAXIS::ROLL:
        p0 = u0;
        p1 = cos(angle)*u1 - sin(angle)*u2;
        p2 = sin(angle)*u1 + cos(angle)*u2;
        break;
      default:
      case RAXIS::YAW:
        p0 = cos(angle)*u0 + sin(angle)*u2;
        p1 = u1;
        p2 = - sin(angle)*u0 + cos(angle)*u2;
        break;
      case RAXIS::PITCH:
        p0 = cos(angle)*u0 - sin(angle)*u1;
        p1 = sin(angle)*u0 + cos(angle)*u1;
        p2 = u2;
        break;
    }
    U(id, 0) = p0 + mean(0);
    U(id, 1) = p1 + mean(1);
    U(id, 2) = p2 + mean(2);
  }
  return U;
}

Eigen::Matrix3d calcTriangleOrientation(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector3d v2, bool calcRightOrientation)
{
    Eigen::Vector3d lead({1, 0, 0});
    Eigen::Vector3d center({0, 1, 0});

    // calculate local triangle from input triangle
    Eigen::Vector3d localv1 = v1 - v0;
    Eigen::Vector3d localv2 = v2 - v0;

/*     // choose hypothenuse/longer side to coincide with the x-axis
    if (localv1.norm() < localv2.norm())
    {
        Eigen::Vector3d temp = localv1;
        localv1 = localv2;
        localv2 = temp;
    } */

    // calculate canonical reference triangle
    // => triangle lies in the xy-plane
    // => triangle vertex v0 lies in the origin (local coordinate system)
    // => triangle vertex v1 lies on the x-axis (edge (v0, v1) coincides with the x-axis)
    // => triangle vertex v2 lies within the first quadrant of the local coordinate system (this sometimes flips triangles, because of the abs() TODO: check if this should be the case or if it does not matter)
    Eigen::Vector3d localv1v2normal = localv1.cross(localv2).normalized();
    Eigen::Vector3d localv1v1v2normal = localv1.cross(localv1v2normal).normalized();
    // double c2y = abs(localv2.dot(localv1v1v2normal));
    double c2y = ((calcRightOrientation) ? 1 : -1) * abs(localv2.dot(localv1v1v2normal));
    // double c2y = ((localv1v2normal[0] >= 0) ? 1 : -1) * abs(localv2.dot(localv1v1v2normal));
    // double c2y = -localv2.dot(localv1v1v2normal);

    // double flip = ((localv1[0] < 0) ? 1 : -1);
    Eigen::Vector3d canonv1 = /* flip * */ localv1.norm()*lead;
    // Eigen::Vector3d canonv1 = localv1.norm()*lead;
    Eigen::Vector3d canonv2 = /* flip * */ (localv1.dot(localv2)/localv1.norm())*lead + c2y*center;

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

Eigen::Matrix3<ADDouble9> calcTinyAD9TriangleOrientation(Eigen::Vector3<ADDouble9> v0, Eigen::Vector3<ADDouble9> v1, Eigen::Vector3<ADDouble9> v2, bool calcRightOrientation)
{
    Eigen::Vector3d lead({1, 0, 0});
    Eigen::Vector3d center({0, 1, 0});

    // calculate local triangle from input triangle
    Eigen::Vector3<ADDouble9> localv1 = v1 - v0;
    Eigen::Vector3<ADDouble9> localv2 = v2 - v0;

/*     // choose hypothenuse/longer side to coincide with the x-axis
    if (localv1.norm() < localv2.norm())
    {
        Eigen::Vector3<ADDouble9> temp = localv1;
        localv1 = localv2;
        localv2 = temp;
    } */

    // calculate canonical reference triangle
    // => triangle lies in the xy-plane
    // => triangle vertex v0 lies in the origin (local coordinate system)
    // => triangle vertex v1 lies on the x-axis (edge (v0, v1) coincides with the x-axis)
    // => triangle vertex v2 lies within the first quadrant of the local coordinate system (this sometimes flips triangles, because of the abs() TODO: check if this should be the case or if it does not matter)
    Eigen::Vector3<ADDouble9> localv1v2normal = localv1.cross(localv2).normalized();
    Eigen::Vector3<ADDouble9> localv1v1v2normal = localv1.cross(localv1v2normal).normalized();
    ADDouble9 c2y = ((calcRightOrientation) ? 1 : -1) * abs(localv2.dot(localv1v1v2normal)); //c2y = abs(c2y);
    // ADDouble9 c2y = ((localv1v2normal[0] >= 0) ? 1 : -1) * abs(localv2.dot(localv1v1v2normal));
    // ADDouble9 c2y = -localv2.dot(localv1v1v2normal); //c2y = abs(c2y);

    // double flip = ((localv1[0] < 0) ? 1 : -1);
    Eigen::Vector3<ADDouble9> canonv1 = /* flip * */ localv1.norm()*lead;
    // Eigen::Vector3<ADDouble9> canonv1 = localv1.norm()*lead;
    Eigen::Vector3<ADDouble9> canonv2 = /* flip * */ (localv1.dot(localv2)/localv1.norm())*lead + c2y*center;

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

// NOTE: triangle points are expected to be the rows of p, q
Eigen::Matrix3d calcTriangleRotation(Eigen::Matrix3d p, Eigen::Matrix3d q)
{
    // center both triangles on their centroids
    Eigen::Matrix3d P = (p.rowwise() - p.colwise().mean());
    Eigen::Matrix3d Q = (q.rowwise() - q.colwise().mean());
    P.transposeInPlace();
    Q.transposeInPlace();
    // Eigen::Matrix3d P = p.rowwise() - p.colwise().mean();
    // Eigen::Matrix3d Q = q.rowwise() - q.colwise().mean();
    // Eigen::Matrix3d P = p.rowwise() - p.row(0);
    // Eigen::Matrix3d Q = q.rowwise() - q.row(0);
    // calculate most rigid rotation between triangles
    Eigen::Matrix3d M = P * Eigen::Matrix3d::Identity() * Q.transpose();
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> UZD = my_svdd(M);
    Eigen::MatrixXd U = std::get<0>(UZD);
    Eigen::MatrixXd S = std::get<1>(UZD);
    Eigen::MatrixXd V = std::get<2>(UZD);

/*     printEigenMatrixXd("p", p);
    printEigenMatrixXd("q", q);
    printEigenMatrixXd("P", P);
    printEigenMatrixXd("Q", Q);
    printEigenMatrixXd("M", M);
    printEigenMatrixXd("singular values", svd.singularValues());
    printEigenMatrixXd("U", svd.matrixU());
    printEigenMatrixXd("V", svd.matrixV()); */
    // return svd.matrixV() * svd.matrixU().transpose();
    return V * Eigen::Matrix3d::Identity() * U.transpose();
    // return U * V.transpose();
}

Eigen::Matrix3<ADDouble9> calcTriangleRotation(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3d q)
{
    // center both triangles on their centroids
    Eigen::Matrix3<ADDouble9> P = (p.rowwise() - p.colwise().mean());
    Eigen::Matrix3d Q = (q.rowwise() - q.colwise().mean());
    P.transposeInPlace();
    Q.transposeInPlace();
/*     Eigen::Matrix3<ADDouble9> P = p.rowwise() - p.row(2);
    Eigen::Matrix3d Q = q.rowwise() - q.row(2); */

    // calculate most rigid rotation between triangles
    Eigen::Matrix3<ADDouble9> M = P * Eigen::Matrix3<ADDouble9>::Identity() * Q.transpose();
    // Eigen::JacobiSVD<Eigen::Matrix3<ADDouble9>> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Eigen::Matrix3<ADDouble9> T;
    // T << 1.07579, 2.07804, 0.000174005,
    //      1.07804, 1.08029, 0.000174368,
    //      0.000174005, 0.000174368, 2.81447e-08;
    // Eigen::MatrixX<ADDouble9> TTT = T.transpose() * T;






    // TODO: svd, evd comparison calculation via eigen
    double xm = (p(0, 0) + p(1, 0) + p(2, 0)).val / 3.0;
    double ym = (p(0, 1) + p(1, 1) + p(2, 1)).val / 3.0;
    double zm = (p(0, 2) + p(1, 2) + p(2, 2)).val / 3.0;
    Eigen::Matrix3d Pr;
    Pr << p(0, 0).val - xm, p(0, 1).val - ym, p(0, 2).val - zm,
          p(1, 0).val - xm, p(1, 1).val - ym, p(1, 2).val - zm,
          p(2, 0).val - xm, p(2, 1).val - ym, p(2, 2).val - zm;
    Pr.transposeInPlace();
    Eigen::Matrix3d Mr = Pr * Eigen::Matrix3d::Identity() * Q.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Mr, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::EigenSolver<Eigen::Matrix3d> evd(Mr.transpose() * Mr);


    // std::complex<ADDouble9> xm = (p(0, 0) + p(1, 0) + p(2, 0)) / 3.0;
    // std::complex<ADDouble9> ym = (p(0, 1) + p(1, 1) + p(2, 1)) / 3.0;
    // std::complex<ADDouble9> zm = (p(0, 2) + p(1, 2) + p(2, 2)) / 3.0;
    // Eigen::Matrix3<std::complex<ADDouble9>> Pcr;
    // Pcr << p(0, 0) - xm, p(0, 1) - ym, p(0, 2) - zm,
    //        p(1, 0) - xm, p(1, 1) - ym, p(1, 2) - zm,
    //        p(2, 0) - xm, p(2, 1) - ym, p(2, 2) - zm;
    // Eigen::Matrix3<std::complex<ADDouble9>> Mcr = Pr.transpose() * Eigen::Matrix3<std::complex<ADDouble9>>::Identity() * Q;

    // Eigen::JacobiSVD<Eigen::Matrix3<std::complex<ADDouble9>>> svd(Mcr, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::EigenSolver<Eigen::Matrix3<std::complex<ADDouble9>>> evd(Mcr.transpose() * Mcr);





    // printTinyAD9Matrix("M", M);

    //std::pair<Eigen::Matrix3<ADDouble9>, Eigen::Vector3<ADDouble9>> jgt = my_jordan_gaussian_transform(M);
    //Eigen::Matrix3<ADDouble9> H = calc_my_hermitian(jgt.second);
    //Eigen::Matrix3<ADDouble9> Hi = calc_my_hermitian_inverse(jgt.second);
    // std::pair<Eigen::Vector3<ADDouble9>, Eigen::Matrix3<ADDouble9>> EVD = my_evd(M);

    std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>> UZD = my_svd(M);
    // std::tuple<Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>, Eigen::MatrixX<ADDouble9>> UZD = my_svd(T);
    Eigen::MatrixX<ADDouble9> U = std::get<0>(UZD);
    Eigen::MatrixX<ADDouble9> S = std::get<1>(UZD);
    Eigen::MatrixX<ADDouble9> V = std::get<2>(UZD);

    // Eigen::MatrixX<ADDouble9> Pp = 
    // for ()

    // std::cout << GREEN;
    // printTinyAD9Matrix("TTT", TTT);
    // std::cout << RESET;
    // std::cout << YELLOW;
    // printTinyAD9Matrix("eigen values", EVD.first);
    // printTinyAD9Matrix("eigen vectors", EVD.second);
    // std::cout << RESET;

/*     printTinyAD9Matrix("jordan gaussian", jgt.first);
    printTinyAD9Matrix("jordan gaussian eigen vector", jgt.second);
    printTinyAD9Matrix("Hermitian", H);
    printTinyAD9Matrix("Hermitian inverse", Hi);
    printTinyAD9Matrix("singular values", std::get<0>(EVD));
    printTinyAD9Matrix("eigen values", std::get<1>(EVD));
    printTinyAD9Matrix("eigen vectors", std::get<2>(EVD)); */


/*     printTinyAD9Matrix("p", p);
    printEigenMatrixXd("q", q);
    printTinyAD9Matrix("P", P);
    printEigenMatrixXd("Q", Q);
    printTinyAD9Matrix("M", M);
    printTinyAD9Matrix("singular values", svd.singularValues());
    printTinyAD9Matrix("U", svd.matrixU());
    printTinyAD9Matrix("V", svd.matrixV()); */
    // printTinyAD9Matrix("M", M);
/*     printTinyAD9Matrix("points p", p);
    printTinyAD9Matrix("points q", q);
    printTinyAD9Matrix("points P", P);
    printEigenMatrixXd("points Pr", Pr);
    printTinyAD9Matrix("points Q", Q); */





    // printTinyAD9Matrix("M", M);
    // printTinyAD9Matrix("MTM", M.transpose() * M);
    // printEigenMatrixXd("Mr", Mr);
    // printEigenMatrixXd("MrTMr", Mr.transpose() * Mr);

    // printEigenMatrixXcd("eigenvalues", evd.eigenvalues().asDiagonal());
    // printEigenMatrixXcd("eigenvectors", evd.eigenvectors());

    // // printTinyAD9Matrix("TTT", TTT);
    // printTinyAD9Matrix("U", U);
    // printTinyAD9Matrix("S", S);
    // printTinyAD9Matrix("V", V);
    // std::cout << CYAN;
    // printTinyAD9Matrix("UTV", U.transpose() * Eigen::Matrix3<ADDouble9>::Identity() * V);
    // std::cout << RESET;
    // std::cout << CYAN;
    // printTinyAD9Matrix("UVT", U * Eigen::Matrix3<ADDouble9>::Identity() * V.transpose());
    // std::cout << RESET;
    // std::cout << CYAN;
    // printTinyAD9Matrix("VTU", V.transpose() * Eigen::Matrix3<ADDouble9>::Identity() * U);
    // std::cout << RESET;
    // std::cout << GREEN;
    // printTinyAD9Matrix("VUT", V * Eigen::Matrix3<ADDouble9>::Identity() * U.transpose());
    // std::cout << RESET;

    // printEigenMatrixXd("Ur", svd.matrixU());
    // printEigenMatrixXd("sr", svd.singularValues().asDiagonal());
    // printEigenMatrixXd("Vr", svd.matrixV());
    // std::cout << GREEN;
    // printEigenMatrixXd("VrUrT", svd.matrixV() * Eigen::Matrix3d::Identity() * svd.matrixU().transpose());
    // std::cout << RESET;
    // printTinyAD9Matrix("points p", p.transpose());
    // printTinyAD9Matrix("points q", q.transpose());
    // printTinyAD9Matrix("points P", P);
    // printTinyAD9Matrix("points Pr", Pr);
    // printTinyAD9Matrix("points Q", Q);
    // printTinyAD9Matrix("points P'", V * Eigen::Matrix3<ADDouble9>::Identity() * U.transpose() * P);

    // printTinyAD9Matrix("points P''", P.col(0).transpose() * V * Eigen::Matrix3<ADDouble9>::Identity() * U.transpose());
    // printTinyAD9Matrix("points P'''", V.transpose() * Eigen::Matrix3<ADDouble9>::Identity() * U * P.col(0));

    // printTinyAD9Matrix("points Pr'", svd.matrixV() * Eigen::Matrix3<ADDouble9>::Identity() * svd.matrixU().transpose() * Pr);





    // printTinyAD9Matrix("points Q'", V * U.transpose() * Q);
    // printTinyAD9Matrix("points Q'", V * U.transpose() * Q);
    // printTinyAD9Matrix("points P''", P * V * U.transpose());
    // printTinyAD9Matrix("points Q''", Q * V * U.transpose());
    // printTinyAD9Matrix("points P'''", U * V.transpose() * P);
    // printTinyAD9Matrix("points Q'''", U * V.transpose() * Q);
    // printTinyAD9Matrix("points P''''", P * U * V.transpose());
    // printTinyAD9Matrix("points Q''''", Q * U * V.transpose());

    //return svd.matrixV() * svd.matrixU().transpose();
    return V * Eigen::Matrix3<ADDouble9>::Identity() * U.transpose();
    // return U * V.transpose();
}

Eigen::Matrix3<ADDouble9> calcTriangleRotation(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3<ADDouble9> q)
{
    // center both triangles on their centroids
    Eigen::Matrix3<ADDouble9> P = p.rowwise() - p.colwise().mean();
    Eigen::Matrix3<ADDouble9> Q = q.rowwise() - q.colwise().mean();
    // calculate most rigid rotation between triangles
    Eigen::Matrix3<ADDouble9> M = P.transpose() * Eigen::Matrix3<ADDouble9>::Identity() * Q;
    Eigen::JacobiSVD<Eigen::Matrix3<ADDouble9>> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // TODO:

    return svd.matrixV() * svd.matrixU().transpose();
}

// NOTE: triangle points are expected to be the rows of p, q (rows 0 are expected to be the vertices that disagree)
double calcTriangleDihedralAngle(Eigen::Matrix3d p, Eigen::Matrix3d q)
{
    Eigen::Vector3d pe1 = p.row(1) - p.row(0);
    Eigen::Vector3d pe2 = p.row(2) - p.row(0);
    Eigen::Vector3d qe1 = q.row(1) - q.row(0);
    Eigen::Vector3d qe2 = q.row(2) - q.row(0);

    Eigen::Vector3d pn = pe1.cross(pe2);
    Eigen::Vector3d qn = qe1.cross(qe2);
    double pnl = abs(pn.norm());
    double qnl = abs(qn.norm());

    // std::cout << "Eigen dihedral " << pn.dot(qn) << std::endl;
    double ndp = pn.dot(qn) / (pnl * qnl);
    if (ndp <= -1) { return M_PI; }
    if (ndp >= 1) { return 0; }
    // if (ndp == -1) { return M_PI; }
    // if (ndp == 1) { return 0; }
    // if (ndp <= -1) { ndp += 0.000000001; }
    return acos(ndp);
}

// NOTE: triangle points are expected to be the rows of p, q (rows 0 are expected to be the vertices that disagree)
ADDouble9 calcTriangleDihedralAngle(Eigen::Matrix3<ADDouble9> p, Eigen::Matrix3d q)
{
    Eigen::Vector3<ADDouble9> pe1 = p.row(1) - p.row(0);
    Eigen::Vector3<ADDouble9> pe2 = p.row(2) - p.row(0);
    Eigen::Vector3d qe1 = q.row(1) - q.row(0);
    Eigen::Vector3d qe2 = q.row(2) - q.row(0);

    Eigen::Vector3<ADDouble9> pn = pe1.cross(pe2);
    Eigen::Vector3d qn = qe1.cross(qe2);
    ADDouble9 pnl = abs(pn.norm());
    double qnl = abs(qn.norm());

    // std::cout << "TinyAD dihedral " << pn.dot(qn).val << std::endl;
    ADDouble9 ndp = pn.dot(qn) / (pnl * qnl);
    if (ndp <= -1) { return M_PI; }
    if (ndp >= 1) { return 0; }
    // if (ndp == -1) { return M_PI; }
    // if (ndp == 1) { return 0; }
    // if (ndp <= -1) { ndp += 0.000000001; }
    // std::cout << "TinyAD calced acos arg: " << ndp.val << std::endl;
    return acos(ndp);
}

double calcEigenSquaredFrobenius(Eigen::Matrix3d matrix)
{
  return matrix(0, 0)*matrix(0, 0) + matrix(0, 1)*matrix(0, 1) + matrix(0, 2)*matrix(0, 2)
       + matrix(1, 0)*matrix(1, 0) + matrix(1, 1)*matrix(1, 1) + matrix(1, 2)*matrix(1, 2)
       + matrix(2, 0)*matrix(2, 0) + matrix(2, 1)*matrix(2, 1) + matrix(2, 2)*matrix(2, 2);
}

ADDouble9 calcTinyAD9SquaredFrobenius(Eigen::Matrix3<ADDouble9> matrix)
{
  return matrix(0, 0)*matrix(0, 0) + matrix(0, 1)*matrix(0, 1) + matrix(0, 2)*matrix(0, 2)
       + matrix(1, 0)*matrix(1, 0) + matrix(1, 1)*matrix(1, 1) + matrix(1, 2)*matrix(1, 2)
       + matrix(2, 0)*matrix(2, 0) + matrix(2, 1)*matrix(2, 1) + matrix(2, 2)*matrix(2, 2);
}

double calcRotationAngle(Eigen::Matrix3d P)
{
    double ht = (P.trace() - 1) / 2;
    if (ht <= -1) { return M_PI; }
    if (ht >= 1) { return 0; }
    return acos(ht);
}

ADDouble9 calcRotationAngle(Eigen::Matrix3<ADDouble9> P)
{
    ADDouble9 ht = (P.trace() - 1) / 2;
    if (ht <= -1) { return M_PI; }
    if (ht >= 1) { return 0; }
    return acos(ht);
}

Eigen::Vector3d calcRotationAxis(Eigen::Matrix3d P)
{
    return Eigen::Vector3d(P(2, 1) - P(1, 2), P(0, 2) - P(2, 0), P(1, 0) - P(0, 1));
}
Eigen::Vector3<ADDouble9> calcRotationAxis(Eigen::Matrix3<ADDouble9> P)
{
    return Eigen::Vector3<ADDouble9>(P(2, 1) - P(1, 2), P(0, 2) - P(2, 0), P(1, 0) - P(0, 1));
}

double calcRotationDifference(Eigen::Matrix3d P, Eigen::Matrix3d Q)
{
    Eigen::Matrix3d R = P * Q.transpose();
    double ht = (R.trace() - 1) / 2;
    if (ht > 1) { return -acos(ht - 1); }
    if (ht < -1) { return acos(ht + 1); }
    // if (ht == -2) { return 2*M_PI; }
    if (ht == -1) { return M_PI; }
    if (ht == 1) { return 0; }
    // if (ht <= -1) { ht += 0.000000001; }
    // if (ht >= 1) { ht -= 0.000000001; }
    // std::cout << std::setprecision(19) << "ht " << ht << std::endl;
    return acos(ht);
}

ADDouble9 calcRotationDifference(Eigen::Matrix3d P, Eigen::Matrix3<ADDouble9> Q)
{
    Eigen::Matrix3<ADDouble9> R = P * Q.transpose();
    ADDouble9 ht = (R.trace() - 1) / 2;
    // std::cout << std::setprecision(10) << "RD1 P trace: " << ((P.trace() - 1) / 2) << std::endl;
    // std::cout << std::setprecision(10) << "RD1 Q trace: " << ((Q.trace() - 1) / 2).val << std::endl;
    // std::cout << CYAN << std::endl;
    // std::cout << std::setprecision(10) << "RD1 ht: " << ht.val << std::endl;
    // std::cout << RESET << std::endl;
    // if (ht > 1) { return -acos(ht - 1); }
    // if (ht < -1) { return acos(ht + 1); }
    // if (ht == -2) { return 2*M_PI; }
    if (ht <= -1) { return M_PI; }
    if (ht >= 1) { return 0; }
    // if (ht <= -1) { ht += 0.000000001; }
    // if (ht >= 1) { ht -= 0.000000001; }
    // std::cout << std::setprecision(19) << "ht " << ht.val << std::endl;
    return acos(ht);
}

ADDouble9 calcRotationDifference(Eigen::Matrix3<ADDouble9> P, Eigen::Matrix3d Q)
{
    Eigen::Matrix3<ADDouble9> R = P * Q.transpose();
    ADDouble9 ht = (R.trace() - 1) / 2;
    if (ht > 1) { return -acos(ht - 1); }
    if (ht < -1) { return acos(ht + 1); }
    // if (ht == -2) { return 2*M_PI; }
    if (ht == -1) { return M_PI; }
    if (ht == 1) { return 0; }
    // if (ht <= -1) { ht += 0.000000001; }
    // if (ht >= 1) { ht -= 0.000000001; }
    // std::cout << std::setprecision(19) << "ht " << ht.val << std::endl;
    return acos(ht);
}

ADDouble9 calcRotationDifference(Eigen::Matrix3<ADDouble9> P, Eigen::Matrix3<ADDouble9> Q)
{
    Eigen::Matrix3<ADDouble9> R = P * Q.transpose();
    ADDouble9 ht = (R.trace() - 1) / 2;
    if (ht > 1) { return -acos(ht - 1); }
    if (ht < -1) { return acos(ht + 1); }
    // if (ht == -2) { return 2*M_PI; }
    if (ht == -1) { return M_PI; }
    if (ht == 1) { return 0; }
    // if (ht <= -1) { ht += 0.000000001; }
    // if (ht >= 1) { ht -= 0.000000001; }
    // std::cout << std::setprecision(19) << "ht " << ht.val << std::endl;
    return acos(ht);
}
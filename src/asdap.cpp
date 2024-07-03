// TODO: implement asdap.hpp
#include "asdap.hpp"

#include <iostream>
#include <fstream>

#include <igl/parallel_for.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/polar_svd3x3.h>

#include <Eigen/Sparse>

typedef Eigen::SparseVector<double>::InnerIterator SVIter;

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data){
  data.V = V;
  data.F = F;
  data.O_inv = std::vector<Eigen::Matrix3d>();
  data.changedV = std::vector<bool>();
  data.nextChangedV = std::vector<bool>();
  for (int i=0; i<V.rows(); i++) { data.changedV.push_back(false); data.nextChangedV.push_back(false); }
  data.hasConverged = false;
  data.iteration = 0;
  data.minimumGradient = 0.00001; // Could also use std::numeric_limits<double>::epsilon
  data.inputMesh.reset(new gcs::ManifoldSurfaceMesh(F));
  data.inputGeometry.reset(new gcs::VertexPositionGeometry(*data.inputMesh, V));
  // TODO: if there are cases where the input mesh is not a triangle mesh, create an intrinsic triangulation of the input mesh (In that case remove the below test)
  // check for non-triangle face and abort, if found
  std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();
  for (int i=0; i<faces.size(); i++)
  {
    if (faces[i].size() != 3)
    {
      throw std::invalid_argument("initMesh: found non-triangle face");
    }
    Eigen::Vector3d v0 = data.V.row(faces[i][0]);
    Eigen::Vector3d v1 = data.V.row(faces[i][1]);
    Eigen::Vector3d v2 = data.V.row(faces[i][2]);
    data.O_inv.push_back(calcTriangleOrientation(v0, v1, v2).inverse());
  }
}

void asdap_set_constraints(const Eigen::VectorXi& constraints_indices, const Eigen::MatrixXd& constraints, ASDAPData& data){
  data.hasConverged = false;
  for (int i=0; i<constraints_indices.size(); i++)
  {
    int index = constraints_indices(i);
    //std::cout << "Enforcing constraint with index = " << index << std::endl;
    float c0 = constraints(i, 0);
    float c1 = constraints(i, 1);
    float c2 = constraints(i, 2);
    if (c0 != c0 || c1 != c1 || c2 != c2)
    {
      std::cout << "found nan constraint: " << constraints(index) << std::endl;
    }
    float p0 = data.V(index, 0);
    float p1 = data.V(index, 1);
    float p2 = data.V(index, 2);
    //std::cout << "Set constraint (" << c0 << ", " << c1 << ", " << c2 << ") for vertex: " << i << std::endl;
    data.changedV[index] = true;
    gcs::Vertex v = data.inputMesh->vertex(index);
    for (gcs::Vertex av : v.adjacentVertices())
    {
      int av_idx = av.getIndex();
      data.changedV[av_idx] = true;
    }
    // std::cout << "set changeV[" << index << "] to true" << std::endl;
  }
}

std::pair<Eigen::MatrixXd, double> asdap_step(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U){
  //std::cout << "--- calculating gradient" << std::endl;
  std::pair<Eigen::MatrixXd, double> result = asdap_facewise_energy_gradient(constraints_indices, data, U);
  //std::cout << "--- Making gradient step" << std::endl;
  // TODO: readd once done debugging/testing
  Eigen::MatrixXd gradients = result.first;
  U = U - lr * gradients;
  data.iteration++;
  //printEigenMatrixXd("gradients", gradients);
  //std::cout << "--- Made gradient step" << std::endl;
  return result;
}

std::pair<Eigen::MatrixXd, double> asdap_optim(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U){
  std::pair<Eigen::MatrixXd, double> result;
  for (int i=0; i<2000 && !data.hasConverged; i++)
  {
    //std::cout << "optimisation iteration: " << data.iteration << std::endl;
    result = asdap_facewise_energy_gradient(constraints_indices, data, U);
    Eigen::MatrixXd gradients = result.first;
    U = U - lr * gradients;
    data.iteration++;
  }
  return result;
}

double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U){
  return asdap_scaling_energy(data, U, ENERGY::SUMOFSQUARES) + asdap_local_rotation_energy(data, U);
}

std::pair<double, double> asdap_energies(const ASDAPData& data, const Eigen::MatrixXd& U){
  return std::pair<double, double>(asdap_scaling_energy(data, U, ENERGY::SUMOFSQUARES), asdap_local_rotation_energy(data, U));
}

// NOTE: Using the determinant formulation leads to incorrect results
double asdap_scaling_energy(const ASDAPData& data, const Eigen::MatrixXd& U, const ENERGY type){
  double result = 0;
  std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();

  Eigen::MatrixXd OG = data.V;
  Eigen::MatrixXd OP = U;

  for (int i=0; i<faces.size(); i++)
  {
    std::vector<size_t> face = faces[i];
    Eigen::Vector3d og_pointp1 = OG.row(face[1]) - OG.row(face[0]);
    Eigen::Vector3d og_pointp2 = OG.row(face[2]) - OG.row(face[0]);

    Eigen::Vector3d op_pointp1 = OP.row(face[1]) - OP.row(face[0]);
    Eigen::Vector3d op_pointp2 = OP.row(face[2]) - OP.row(face[0]);

    double og_area = og_pointp1.cross(og_pointp2).norm();

    Eigen::Vector2d lead = Eigen::Vector2d({1, 0});
    Eigen::Vector2d trail = Eigen::Vector2d({0, 1});

    // project triangle into 2d
    double og2d_p1x = og_pointp1.norm();
    double og2d_p2x = og_pointp1.dot(og_pointp2) / og2d_p1x;
    double og2d_p2y = sqrt(og_pointp2.squaredNorm() - og2d_p2x * og2d_p2x);
    Eigen::Vector2d og2d_p1 = og2d_p1x * lead;
    Eigen::Vector2d og2d_p2 = og2d_p2x * lead + og2d_p2y * trail;

    double op2d_p1x = op_pointp1.norm();
    double op2d_p2x = op_pointp1.dot(op_pointp2) / op2d_p1x;
    double op2d_p2y = sqrt(op_pointp2.squaredNorm() - op2d_p2x * op2d_p2x);
    Eigen::Vector2d op2d_p1 = op2d_p1x * lead;
    Eigen::Vector2d op2d_p2 = op2d_p2x * lead + op2d_p2y * trail;

    Eigen::Matrix2d Og(2, 2);
    Eigen::Matrix2d Op(2, 2);

    Og = og2d_p1*lead.transpose() + og2d_p2*trail.transpose();
    Op = op2d_p1*lead.transpose() + op2d_p2*trail.transpose();
    Eigen::Matrix2d Og_inv = Og.inverse();

    Op.applyOnTheRight(Og_inv);
    Eigen::Matrix2d T = Op;
    Eigen::Matrix2d T_inv = T.inverse();

    double det, det_inv;
    double anti_sym_updownscaling, anti_shearing;
    switch(type)
    {
      case ENERGY::DETERMINANT:
        det = T.determinant();
        det_inv = T_inv.determinant();
        result += og_area*(det*det + det_inv*det_inv);
        break;
      case ENERGY::SUMOFSQUARES:
        anti_sym_updownscaling = T(0,0)*T(0,0) + T(1, 1)*T(1,1) + T_inv(0,0)*T_inv(0,0) + T_inv(1, 1)*T_inv(1,1);
        anti_shearing = T(0,1)*T(0,1) + T(1, 0)*T(1,0) + T_inv(0,1)*T_inv(0,1) + T_inv(1, 0)*T_inv(1,0);
        result += og_area*(anti_sym_updownscaling + anti_shearing);
       break;
    }
  }

  return result;
}

double asdap_local_rotation_energy(const ASDAPData& data, const Eigen::MatrixXd& U){
  double energy = 0.0;
  for (int i=0; i<data.F.rows(); i++)
  {
    int face_idx = i;
    int v0_idx = data.F(i, 0);
    int v1_idx = data.F(i, 1);
    int v2_idx = data.F(i, 2);
    Eigen::Vector3d v0 = U.row(v0_idx);
    Eigen::Vector3d v1 = U.row(v1_idx);
    Eigen::Vector3d v2 = U.row(v2_idx);

    Eigen::Matrix3d orientation = calcTriangleOrientation(v0, v1, v2);

    //std::cout << "FACE: face" << face_idx << std::endl;
    gcs::Face gcsFace = data.inputMesh->face(face_idx);
    for (gcs::Face gcsNeighbour : gcsFace.adjacentFaces())
    {
      int neighbourIndex = gcsNeighbour.getIndex();
      Eigen::Vector3i neighbour_eigen_face = data.F.row(neighbourIndex);
      Eigen::Matrix3d neighbour_orientation = calcTriangleOrientation(U.row(neighbour_eigen_face(0)), U.row(neighbour_eigen_face(1)), U.row(neighbour_eigen_face(2)));

      // calculate original rotation between adjacent faces
      Eigen::Matrix3d og_rotation = data.O_inv[face_idx].inverse()*data.O_inv[neighbourIndex];
      // calculate current rotation between adjacent faces
      Eigen::Matrix3d op_rotation = orientation*neighbour_orientation.inverse();
      // subtract current rotation between adjacent faces from original rotation between them
      Eigen::Matrix3d rotation_diff = og_rotation - op_rotation;
      energy += calcEigenSquaredFrobenius(rotation_diff);
    }
  }
  // TODO: this was just testing if weird behaviour can be reduced by reducing the gradient
  energy /= 2.0;

  // return the aggregate of the local face rotation energy
  return energy;

}

// TODO: idk what the heck is going on here. Random crashes. Can't find the source... Seems to come from the gcs data structure neighbourhood queries... (It was too much drawing for the libigl viewer to handle)
// TODO: could change face to sport size_t indices instead of integers.
std::pair<Eigen::Matrix3d, double> asdap_energy_face_scaling_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face){
  //std::cout << "pre face" << std::endl;
  //std::cout << "Checking face " << face[0] << ", " << face[1] << ", " << face[2] << std::endl;
  //std::cout << "pre gradient" << std::endl;
  Eigen::Matrix3d gradient;
  double energy = 0.0;

  //std::cout << "pre make active" << std::endl;
  // get all face vertices
  Eigen::MatrixXd OG = data.V;
  Eigen::MatrixXd OP = U;
  Eigen::VectorX<ADDouble9> verts = ADDouble9::make_active({OP(face[0], 0), OP(face[0], 1), OP(face[0], 2), OP(face[1], 0), OP(face[1], 1), OP(face[1], 2), OP(face[2], 0), OP(face[2], 1), OP(face[2], 2)}); // TODO: check if these are ordered right

  //std::cout << "pre select" << std::endl;
  Eigen::MatrixXd select_first_point(3, 9);
  Eigen::MatrixXd select_second_point(3, 9);
  Eigen::MatrixXd select_third_point(3, 9);
  select_first_point << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero();
  select_second_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
  select_third_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity();

  Eigen::Vector3<ADDouble9> opp0 = select_first_point * verts;
  Eigen::Vector3<ADDouble9> opp1 = select_second_point * verts;
  Eigen::Vector3<ADDouble9> opp2 = select_third_point * verts;

  //std::cout << "pre points" << std::endl;
  Eigen::Vector3d og_pointp1;
  Eigen::Vector3d og_pointp2;
  og_pointp1 = OG.row(face[1]) - OG.row(face[0]);
  og_pointp2 = OG.row(face[2]) - OG.row(face[0]);
  Eigen::Vector3<ADDouble9> op_pointp1;
  Eigen::Vector3<ADDouble9> op_pointp2;
  op_pointp1 = opp1 - opp0;
  op_pointp2 = opp2 - opp0;

  //std::cout << "pre og_area" << std::endl;
  double og_area = og_pointp1.cross(og_pointp2).norm();

  Eigen::Vector2d lead = Eigen::Vector2d({1, 0});
  Eigen::Vector2d trail = Eigen::Vector2d({0, 1});

  Eigen::Vector2<ADDouble9> op_lead = Eigen::Vector2<ADDouble9>({1, 0});
  Eigen::Vector2<ADDouble9> op_trail = Eigen::Vector2<ADDouble9>({0, 1});

  //std::cout << "pre helper vals" << std::endl;
  double og2d_p1x = og_pointp1.norm();
  double og2d_p2x = og_pointp1.dot(og_pointp2) / og2d_p1x;
  double og2d_p2y = sqrt(og_pointp2.squaredNorm() - og2d_p2x * og2d_p2x);
  Eigen::Vector2d og2d_p1 = og2d_p1x * lead;
  Eigen::Vector2d og2d_p2 = og2d_p2x * lead + og2d_p2y * trail;

  //std::cout << "pre points 2d" << std::endl;
  ADDouble9 op2d_p1x = op_pointp1.norm();
  ADDouble9 op2d_p2x = op_pointp1.dot(op_pointp2) / op2d_p1x;
  ADDouble9 op2d_p2y = sqrt(op_pointp2.squaredNorm() - op2d_p2x * op2d_p2x);
  Eigen::Vector2<ADDouble9> op2d_p1 = op2d_p1x * op_lead;
  Eigen::Vector2<ADDouble9> op2d_p2 = op2d_p2x * op_lead + op2d_p2y * op_trail;

  //std::cout << "pre Og, Op" << std::endl;
  Eigen::Matrix2d Og(2, 2);
  Eigen::Matrix2<ADDouble9> Op(2, 2);

  Og = og2d_p1*lead.transpose() + og2d_p2*trail.transpose();
  Op = op2d_p1*op_lead.transpose() + op2d_p2*op_trail.transpose();
  Eigen::Matrix2d Og_inv = Og.inverse();

  //std::cout << "pre T, T_inv" << std::endl;
  Op.applyOnTheRight(Og_inv);
  Eigen::Matrix2<ADDouble9> T = Op;
  Eigen::Matrix2<ADDouble9> T_inv = T.inverse();
  //printTinyAD9Matrix("T", T);
  //printTinyAD9Matrix("TINV", T_inv);

  //std::cout << "pre dets" << std::endl;
/*   ADDouble9 det = T.determinant();
  ADDouble9 det_inv = T_inv.determinant(); */

  //std::cout << "pre result terms" << std::endl;
  ADDouble9 anti_sym_updownscaling = T(0,0)*T(0,0) + T(1,1)*T(1,1);
  ADDouble9 inv_anti_sym_updownscaling = T_inv(0,0)*T_inv(0,0) + T_inv(1,1)*T_inv(1,1);
  ADDouble9 anti_shearing = T(0,1)*T(0,1) + T(1,0)*T(1,0);
  ADDouble9 inv_anti_shearing = T_inv(0,1)*T_inv(0,1) + T_inv(1,0)*T_inv(1,0);

  //std::cout << "pre results" << std::endl;
  //std::cout << "anti_sym_UPDOWN " << anti_sym_updownscaling.val << std::endl;
  //std::cout << "inv_anti_sym_UPDOWN " << inv_anti_sym_updownscaling.val << std::endl;
  //std::cout << "anti_shear " << anti_shearing.val << std::endl;
  //std::cout << "inv_anti_shear " << inv_anti_shearing.val << std::endl;
  // TODO: figure out solution to exploding gradients when transformation matrix T has miniscule values and thus T_inv has ginormous ones
  //ADDouble9 determinant_energy = og_area*(det*det + det_inv*det_inv);
  ADDouble9 result = og_area*(anti_sym_updownscaling + anti_shearing + inv_anti_sym_updownscaling + inv_anti_shearing);

  //std::cout << "pre energy_sum" << std::endl;
  //std::cout << "energy_sum: " << energy_sum << ", energy.val: " << energy.val << std::endl;
  energy += result.val;
  //std::cout << "pre gradient push" << std::endl;
  gradient << result.grad(0), result.grad(1), result.grad(2),
              result.grad(3), result.grad(4), result.grad(5),
              result.grad(6), result.grad(7), result.grad(8);

  return std::pair<Eigen::Matrix3d, double>(gradient, energy);
}

std::pair<Eigen::Matrix3d, double> asdap_energy_local_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx)
{
  Eigen::Matrix3d gradient;
  double energy = 0.0;
  ADDouble9 result = 0.0;

  Eigen::VectorX<ADDouble9> verts = ADDouble9::make_active({U(face[0], 0), U(face[0], 1), U(face[0], 2), U(face[1], 0), U(face[1], 1), U(face[1], 2), U(face[2], 0), U(face[2], 1), U(face[2], 2)});

  Eigen::MatrixXd select_first_point(3, 9);
  Eigen::MatrixXd select_second_point(3, 9);
  Eigen::MatrixXd select_third_point(3, 9);
  select_first_point << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero();
  select_second_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
  select_third_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity();

  Eigen::Vector3<ADDouble9> v0 = select_first_point * verts;
  Eigen::Vector3<ADDouble9> v1 = select_second_point * verts;
  Eigen::Vector3<ADDouble9> v2 = select_third_point * verts;

  Eigen::Matrix3<ADDouble9> orientation = calcTinyAD9TriangleOrientation(v0, v1, v2);

  //std::cout << "FACE: face" << face_idx << std::endl;
  gcs::Face gcsFace = data.inputMesh->face(face_idx);
  for (gcs::Face gcsNeighbour : gcsFace.adjacentFaces())
  {
    int neighbourIndex = gcsNeighbour.getIndex();
    Eigen::Vector3i neighbour_eigen_face = data.F.row(neighbourIndex);
    Eigen::Matrix3d neighbour_orientation = calcTriangleOrientation(U.row(neighbour_eigen_face(0)), U.row(neighbour_eigen_face(1)), U.row(neighbour_eigen_face(2)));

    // calculate original rotation between adjacent faces
    Eigen::Matrix3<ADDouble9> og_rotation = data.O_inv[face_idx].inverse()*data.O_inv[neighbourIndex];
    // calculate current rotation between adjacent faces
    Eigen::Matrix3<ADDouble9> op_rotation = orientation*neighbour_orientation.inverse();
    // subtract current rotation between adjacent faces from original rotation between them
    Eigen::Matrix3<ADDouble9> rotation_diff = og_rotation - op_rotation;

    result += calcTinyAD9SquaredFrobenius(rotation_diff);
    energy += result.val;

/*     std::cout << " => neighbour: face" << neighbourIndex << " (Energy: " << result.val << ")" << std::endl;
    printTinyAD9Matrix("og_rotation", og_rotation);
    printTinyAD9Matrix("op_rotation", op_rotation);
    printTinyAD9Matrix("rotation_difference", rotation_diff); */
  }

  result /= 2.0;
  gradient << result.grad(0), result.grad(1), result.grad(2),
              result.grad(3), result.grad(4), result.grad(5),
              result.grad(6), result.grad(7), result.grad(8);

  // return the aggregate of the vertex gradients
  return std::pair<Eigen::Matrix3d, double>(gradient, energy);
}

std::pair<Eigen::Matrix3d, double> asdap_energy_global_face_rotation_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<int>& face, const int face_idx, const Eigen::Matrix3d avg_rotation)
{
  Eigen::Matrix3d gradient;
  double energy = 0.0;

  Eigen::VectorX<ADDouble9> verts = ADDouble9::make_active({U(face[0], 0), U(face[0], 1), U(face[0], 2), U(face[1], 0), U(face[1], 1), U(face[1], 2), U(face[2], 0), U(face[2], 1), U(face[2], 2)}); // TODO: check if these are ordered right

  Eigen::MatrixXd select_first_point(3, 9);
  Eigen::MatrixXd select_second_point(3, 9);
  Eigen::MatrixXd select_third_point(3, 9);
  select_first_point << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero();
  select_second_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
  select_third_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity();

  Eigen::Vector3<ADDouble9> v0 = select_first_point * verts;
  Eigen::Vector3<ADDouble9> v1 = select_second_point * verts;
  Eigen::Vector3<ADDouble9> v2 = select_third_point * verts;

  Eigen::Matrix3<ADDouble9> orientation = calcTinyAD9TriangleOrientation(v0, v1, v2);

  Eigen::Matrix3<ADDouble9> rotation = orientation*data.O_inv[face_idx];

  Eigen::Matrix3<ADDouble9> rot_diff = avg_rotation - rotation;
  ADDouble9 result = calcTinyAD9SquaredFrobenius(rot_diff);

  energy = result.val;
  gradient << result.grad(0), result.grad(1), result.grad(2),
              result.grad(3), result.grad(4), result.grad(5),
              result.grad(6), result.grad(7), result.grad(8);


/*   printEigenMatrixXd("O_inv", data.O_inv[face_idx]);
  printTinyAD9Matrix("orientation", orientation);
  printTinyAD9Matrix("rotation", rotation);
  printEigenMatrixXd("avg_rotation", avg_rotation);
  printTinyAD9Matrix("rot_diff", rot_diff); */

  // return the aggregate of the vertex gradients
  return std::pair<Eigen::Matrix3d, double>(gradient, energy);
}

Eigen::Matrix3d calcAvgRotation(const Eigen::VectorXi& constraints_indices, const ASDAPData& data, const Eigen::MatrixXd& U)
{
  // TODO: this way of averaging, does not really work
  Eigen::Matrix3d average;
  int i=0;
  for (; i<data.F.rows(); i++)
  {
    Eigen::Matrix3d orientation = calcTriangleOrientation(U.row(data.F(i, 0)), U.row(data.F(i, 1)), U.row(data.F(i, 2)));
    Eigen::Matrix3d rotation = orientation*data.O_inv[i];
    average += rotation;

/*     std::cout << "Matrices in avg calculation iteration " << i << std::endl;
    printEigenMatrixXd("v0", U.row(data.F(i, 0)));
    printEigenMatrixXd("v1", U.row(data.F(i, 1)));
    printEigenMatrixXd("v2", U.row(data.F(i, 2)));
    printEigenMatrixXd("orientation", orientation);
    printEigenMatrixXd("O_inv", data.O_inv[i]);
    printEigenMatrixXd("current average", average); */
  }

  return average/i;
}

// Face based
std::pair<Eigen::MatrixXd, double> asdap_facewise_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U)
{
  int UR = U.rows();
  int UC = U.cols();
  Eigen::MatrixXd gradients = Eigen::MatrixXd::Zero(UR, UC);
  double energy = 0.0;
  bool foundGradient = false;

  // create isVertexConstrained lookup table for efficiency
  std::vector<bool> isVertexConstrained = std::vector<bool>(data.V.rows());
  for (int i=0; i<data.V.rows(); i++) { isVertexConstrained[i] = false; }
  for (int i=0; i<constraints_indices.size(); i++) { isVertexConstrained[constraints_indices(i)] = true; }

  // TODO: remove, when its clear global bending term is not needed
  Eigen::Matrix3d avg_rot = calcAvgRotation(constraints_indices, data, U);

  // std::cout << "Found number of vertices: " << data.V.rows() << std::endl;
  // iterate over all faces
  for (int i=0; i<data.F.rows(); i++)
  {
    int v0 = data.F(i, 0);
    int v1 = data.F(i, 1);
    int v2 = data.F(i, 2);
    //std::cout << "------ Checking face" << i << " with vertices " << v0 << ", " << v1 << ", " << v2 << std::endl;
    // skip constrained vertices & non-changed vertices
    // TODO: because of changed optimisation energy becomes lower (fix by always saving the last energy value for each vertex & adding it to the energy here)
    bool skipv0 = isVertexConstrained[v0] /* || !data.changedV[v0] */;
    bool skipv1 = isVertexConstrained[v1] /* || !data.changedV[v1] */;
    bool skipv2 = isVertexConstrained[v2] /* || !data.changedV[v2] */;
    if (skipv0 && skipv1 && skipv2) {
      //std::cout << "------> vertices " << v0 << ", " << v1 << ", " << v2 << " are constrained or unchanged." << std::endl;
      //std::cout << "------> skipping face" << i << std::endl;
      continue;
    }

    //std::cout << "------ Calculating gradient for face" << i << std::endl;

    // collect the energy & gradient for each vertex of a face with respect to that face
    std::vector<int> face = std::vector<int>({v0, v1, v2});
    std::pair<Eigen::Matrix3d, double> result0 = asdap_energy_face_scaling_gradients(data, U, face);
    std::pair<Eigen::Matrix3d, double> result1 = asdap_energy_local_face_rotation_gradients(data, U, face, i);
    //std::cout << "Calculated gradients: (scaling) " << result0.second << ", (rotation) " << result1.second << std::endl;
    //std::cout << "------ aggregating gradients" << std::endl;
    Eigen::Vector3d v0_grad = result0.first.row(0) + result1.first.row(0);
    Eigen::Vector3d v1_grad = result0.first.row(1) + result1.first.row(1);
    Eigen::Vector3d v2_grad = result0.first.row(2) + result1.first.row(2);
    if (v0_grad != v0_grad || v1_grad != v1_grad || v2_grad != v2_grad) {
      std::cout << "Found NAN gradient in vertices: v" << v0 << ": " << (v0_grad != v0_grad) << ", "
                                                   "v" << v1 << ": " << (v1_grad != v1_grad) << ", "
                                                   "v" << v2 << ": " << (v2_grad != v2_grad) << std::endl;
    }
    if (!skipv0) { gradients(v0, 0) += v0_grad(0); gradients(v0, 1) += v0_grad(1); gradients(v0, 2) += v0_grad(2); }
    if (!skipv1) { gradients(v1, 0) += v1_grad(0); gradients(v1, 1) += v1_grad(1); gradients(v1, 2) += v1_grad(2); }
    if (!skipv2) { gradients(v2, 0) += v2_grad(0); gradients(v2, 1) += v2_grad(1); gradients(v2, 2) += v2_grad(2); }
    energy += result0.second + result1.second;

    //std::cout << "------ does this neighbourhood remain active: ";
    std::vector<Eigen::Vector3d> grads = std::vector<Eigen::Vector3d>({v0_grad, v1_grad, v2_grad});
    for (int i=0; i<3; i++)
    {
      int v = face[i];
      Eigen::Vector3d v_grad = grads[i];
      if (v_grad.norm() > data.minimumGradient)
      {
        //std::cout << "Yes";
        foundGradient = true;
        data.nextChangedV[v] = true;
        gcs::Vertex gcsv = data.inputMesh->vertex(v);
        for (gcs::Vertex av : gcsv.adjacentVertices())
        {
          int av_idx = av.getIndex();
          data.nextChangedV[av_idx] = true;
        }
      }
      //std::cout << std::endl;
    }
  }

  //std::cout << "--- finished gradient calculation. Setting hasConverged = " << !foundGradient << " and changedV" << std::endl;
  data.hasConverged = !foundGradient;
  data.changedV = data.nextChangedV;
  data.nextChangedV = std::vector<bool>();
  for (int i=0; i<data.V.rows(); i++) { data.nextChangedV.push_back(false); }
  // return the aggregate of the vertex gradients
  return std::pair<Eigen::MatrixXd, double>(gradients, energy);
}
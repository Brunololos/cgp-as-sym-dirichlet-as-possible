// TODO: implement asdap.hpp
#include "asdap.hpp"

#include <iostream>
#include <fstream>

#include <igl/parallel_for.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/polar_svd3x3.h>

#include <Eigen/Sparse>

#include "asdap_utils.hpp"

typedef Eigen::SparseVector<double>::InnerIterator SVIter;

void initMesh(const Eigen::Matrix<double, -1, 3>& V, const Eigen::Matrix<int, -1, 3>& F, ASDAPData& data){
  data.V = V;
  data.F = F;
  data.changedV = std::vector<bool>();
  data.nextChangedV = std::vector<bool>();
  for (int i=0; i<V.rows(); i++) { data.changedV.push_back(false); data.nextChangedV.push_back(false); }
  data.hasConverged = false;
  data.iteration = 0;
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
  }
}

void asdap_precompute(const Eigen::VectorXi& constraints_indices, const Eigen::MatrixXd& constraints, ASDAPData& data){
  //data.V = data.inputGeometry->vertexPositions; // TODO: reset vertex positions to initial positions? Maybe we want to edit during the optimisation procedure
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
/*     if (p0 != c0 || p1 != c1 || p2 != c2)
    {
      std::cout << "found case where vertex position is changed: (" << p0 << ", " << p1 << ", " << p2 << ") => (" << c0 << ", " << c1 << ", " << c2 << ")" << std::endl;
    } */
    //std::cout << "Setting constraint for vertex " + std::to_string(index) << std::endl;
    //data.V(index) = constraints(i);
    // This is forbidden don't change the original vertices
/*     data.V(index, 0) = c0;
    data.V(index, 1) = c1;
    data.V(index, 2) = c2; */
    data.changedV[index] = true;
    gcs::Vertex v = data.inputMesh->vertex(index);
    for (gcs::Vertex av : v.adjacentVertices())
    {
      int av_idx = av.getIndex();
      data.changedV[av_idx] = true;
    }
    //std::cout << "set changeV[" << index << "] to true" << std::endl;
  }
}

std::pair<Eigen::MatrixXd, double> asdap_step(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U){
  std::cout << "--- calculating gradient" << std::endl;
  std::pair<Eigen::MatrixXd, double> result = asdap_energy_gradient(constraints_indices, data, U);
  std::cout << "--- Making gradient step" << std::endl;
  Eigen::MatrixXd gradients = result.first;
  U = U - lr * gradients;
  data.iteration++;
  return result;
}

std::pair<Eigen::MatrixXd, double> asdap_optim(const Eigen::VectorXi& constraints_indices, double lr, ASDAPData& data, Eigen::MatrixXd& U){
  std::pair<Eigen::MatrixXd, double> result;
  for (int i=0; i<2000 && !data.hasConverged; i++)
  {
    std::cout << "optimisation iteration: " << data.iteration << std::endl;
    result = asdap_energy_gradient(constraints_indices, data, U);
    Eigen::MatrixXd gradients = result.first;
    U = U - lr * gradients;
    data.iteration++;
  }
  return result;
}

// TODO: this energy still uses the determinant formulation
double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U){
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

    float og_area = og_pointp1.cross(og_pointp2).norm(); // TODO: this is double the triangle area. But we don't care much about adding an additional 1/2 factor.
    Eigen::Vector3d lead = Eigen::Vector3d({1, 0, 0});
    Eigen::Vector3d center = Eigen::Vector3d({0, 1, 0});
    Eigen::Vector3d trail = Eigen::Vector3d({0, 0, 1});

    // TODO: instead of constructing 3x3 matrices and determining their determinants, it might be more efficient to project the triangles into 2d space and calculate the determinants there. As we are only interested in the determinant values, we dont even care to preserve rotation or translation.
    Eigen::Vector3d og_orthogonal = og_pointp1.cross(og_pointp2).normalized();
    Eigen::Vector3d op_orthogonal = op_pointp1.cross(op_pointp2).normalized();

    Eigen::Matrix3d Og(3, 3);
    Eigen::Matrix3d Op(3, 3);
    Og = og_orthogonal*lead.transpose() + og_pointp1*center.transpose() + og_pointp2*trail.transpose();
    Op = op_orthogonal*lead.transpose() + op_pointp1*center.transpose() + op_pointp2*trail.transpose();

    Eigen::Matrix3d Og_inv = Og.inverse();

    Op.applyOnTheRight(Og_inv);
    Eigen::Matrix3d T = Op;
    Eigen::Matrix3d T_inv = T.inverse();

    double det = T.determinant();
    double det_inv = T_inv.determinant();

    // TODO: add bending term
    result += og_area*(det*det + det_inv*det_inv);
/*     if (fabs(det - 1.0f) < std::numeric_limits<float>::epsilon() || fabs(det_inv - 1.0f) < std::numeric_limits<float>::epsilon()) {
      std::cout << "found transformed triangle with det = " + std::to_string(det) +" & det_inv = " + std::to_string(det_inv) << std::endl;
      std::cout << "Double triangle area = " + std::to_string(og_area) << std::endl;
    } */
    //std::cout << "Intermediate ASDAP energy: " + std::to_string(result) << std::endl;
    if ( result != result) {
      std::cout << "Face " + std::to_string(i) + ": (original) <" + std::to_string(OG(face[0], 0)) + ", "
                                                                  + std::to_string(OG(face[0], 1)) + ", "
                                                                  + std::to_string(OG(face[0], 2)) + ">, "
                                                            + "<" + std::to_string(OG(face[1], 0)) + ", "
                                                                  + std::to_string(OG(face[1], 1)) + ", "
                                                                  + std::to_string(OG(face[1], 2)) + ">, "
                                                            + "<" + std::to_string(OG(face[2], 0)) + ", "
                                                                  + std::to_string(OG(face[2], 1)) + ", "
                                                                  + std::to_string(OG(face[2], 2)) + ">" << std::endl;
      std::cout << "Face " + std::to_string(i) + ": (optimised) <" + std::to_string(OP(face[0], 0)) + ", "
                                                                  + std::to_string(OP(face[0], 1)) + ", "
                                                                  + std::to_string(OP(face[0], 2)) + ">, "
                                                            + "<" + std::to_string(OP(face[1], 0)) + ", "
                                                                  + std::to_string(OP(face[1], 1)) + ", "
                                                                  + std::to_string(OP(face[1], 2)) + ">, "
                                                            + "<" + std::to_string(OP(face[2], 0)) + ", "
                                                                  + std::to_string(OP(face[2], 1)) + ", "
                                                                  + std::to_string(OP(face[2], 2)) + ">" << std::endl;
      std::cout << "og_pointp1:\n" << og_pointp1 << std::endl;
      std::cout << "og_pointp2:\n" << og_pointp2 << std::endl;
      std::cout << "op_pointp1:\n" << op_pointp1 << std::endl;
      std::cout << "op_pointp2:\n" << op_pointp2 << std::endl;

      std::cout << "OG, OP:" << std::endl;
      std::cout << Og << std::endl;
      std::cout << Op << std::endl;
      std::cout << "OGINV:" << std::endl;
      std::cout << Og.inverse() << std::endl;

      std::cout << "T = OP x OGINV:" << std::endl;
      std::cout << T << std::endl;

      std::cout << "T det:" << std::endl;
      std::cout << det << std::endl;

      std::cout << "TINV det:" << std::endl;
      std::cout << det_inv << std::endl;
      std::cout << "Double triangle area = " + std::to_string(og_area) << std::endl;
      return result;
    }
  }

  return result;
}

// TODO: idk what the heck is going on here. Random crashes. Can't find the source...
std::pair<Eigen::Matrix3d, double> asdap_energy_face_vertex_gradients(const ASDAPData& data, const Eigen::MatrixXd& U, const std::vector<size_t>& face){
  std::cout << "pre face" << std::endl;
/*   std::cout << data.inputMesh->getFaceVertexList()[face_idx][0] << std::endl;
  std::cout << data.inputMesh->getFaceVertexList()[face_idx][1] << std::endl;
  std::cout << data.inputMesh->getFaceVertexList()[face_idx][2] << std::endl; */
  std::cout << face[0] << ", " << face[1] << ", " << face[2] << std::endl;
/*   std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();
  std::cout << "got face list" << std::endl; */
  //Eigen::Vector3i face = data.F.row(face_idx);//data.inputMesh->getFaceVertexList().at(face_idx);
/*   std::cout << data.F.row(face_idx)[0] << std::endl;
  std::cout << data.F.row(face_idx)[1] << std::endl;
  std::cout << data.F.row(face_idx)[2] << std::endl; */
  std::cout << "pre gradient" << std::endl;
  Eigen::Matrix3d gradient/*  = (Eigen::Matrix3d() << 0, 0, 0, 0, 0, 0, 0, 0, 0).finished() */;
  double result_sum = 0.0;

  std::cout << "pre make active" << std::endl;
  // get all face vertices
  Eigen::MatrixXd OG = data.V;
  Eigen::MatrixXd OP = U;
  Eigen::VectorX<ADDouble9> verts = ADDouble9::make_active({OP(face[0], 0), OP(face[0], 1), OP(face[0], 2), OP(face[1], 0), OP(face[1], 1), OP(face[1], 2), OP(face[2], 0), OP(face[2], 1), OP(face[2], 2)}); // TODO: check if these are ordered right

  std::cout << "pre select" << std::endl;
  Eigen::MatrixXd select_first_point(3, 9);
  Eigen::MatrixXd select_second_point(3, 9);
  Eigen::MatrixXd select_third_point(3, 9);
  select_first_point << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero();
  select_second_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
  select_third_point << Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity();

  Eigen::Vector3<ADDouble9> opp0 = select_first_point * verts;
  Eigen::Vector3<ADDouble9> opp1 = select_second_point * verts;
  Eigen::Vector3<ADDouble9> opp2 = select_third_point * verts;

  std::cout << "pre points" << std::endl;
  Eigen::Vector3d og_pointp1;
  Eigen::Vector3d og_pointp2;
  og_pointp1 = OG.row(face[1]) - OG.row(face[0]);
  og_pointp2 = OG.row(face[2]) - OG.row(face[0]);
  Eigen::Vector3<ADDouble9> op_pointp1;
  Eigen::Vector3<ADDouble9> op_pointp2;
  op_pointp1 = opp1 - opp0;
  op_pointp2 = opp2 - opp0;

  std::cout << "pre og_area" << std::endl;
  double og_area = og_pointp1.cross(og_pointp2).norm();

  Eigen::Vector2d lead = Eigen::Vector2d({1, 0});
  Eigen::Vector2d trail = Eigen::Vector2d({0, 1});

  Eigen::Vector2<ADDouble9> op_lead = Eigen::Vector2<ADDouble9>({1, 0});
  Eigen::Vector2<ADDouble9> op_trail = Eigen::Vector2<ADDouble9>({0, 1});

  std::cout << "pre helper vals" << std::endl;
  double og2d_p1x = og_pointp1.norm();
  double og2d_p2x = og_pointp1.dot(og_pointp2) / og2d_p1x;
  double og2d_p2y = sqrt(og_pointp2.squaredNorm() - og2d_p2x * og2d_p2x);
  Eigen::Vector2d og2d_p1 = og2d_p1x * lead;
  Eigen::Vector2d og2d_p2 = og2d_p2x * lead + og2d_p2y * trail;

  std::cout << "pre points 2d" << std::endl;
  ADDouble9 op2d_p1x = op_pointp1.norm();
  ADDouble9 op2d_p2x = op_pointp1.dot(op_pointp2) / op2d_p1x;
  ADDouble9 op2d_p2y = sqrt(op_pointp2.squaredNorm() - op2d_p2x * op2d_p2x);
  Eigen::Vector2<ADDouble9> op2d_p1 = op2d_p1x * op_lead;
  Eigen::Vector2<ADDouble9> op2d_p2 = op2d_p2x * op_lead + op2d_p2y * op_trail;

  std::cout << "pre Og, Op" << std::endl;
  Eigen::Matrix2d Og(2, 2);
  Eigen::Matrix2<ADDouble9> Op(2, 2);

  Og = og2d_p1*lead.transpose() + og2d_p2*trail.transpose();
  Op = op2d_p1*op_lead.transpose() + op2d_p2*op_trail.transpose();
  Eigen::Matrix2d Og_inv = Og.inverse();

  std::cout << "pre T, T_inv" << std::endl;
  Eigen::MatrixX<ADDouble9> saveOp = Op; // TODO: delete later ONLY USED FOR PRINTS
  Op.applyOnTheRight(Og_inv);
  Eigen::Matrix2<ADDouble9> T = Op;
  Eigen::Matrix2<ADDouble9> T_inv = T.inverse();

  std::cout << "pre dets" << std::endl;
  ADDouble9 det = T.determinant();
  ADDouble9 det_inv = T_inv.determinant();

  std::cout << "pre result terms" << std::endl;
  ADDouble9 anti_sym_updownscaling = T(0,0)*T(0,0) + T(1, 1)*T(1,1)
                                    + T_inv(0,0)*T_inv(0,0) + T_inv(1, 1)*T_inv(1,1);
  ADDouble9 anti_shearing = T(0,1)*T(0,1) + T(1, 0)*T(1,0)
                                    + T_inv(0,1)*T_inv(0,1) + T_inv(1, 0)*T_inv(1,0);

  std::cout << "pre results" << std::endl;
  std::cout << "anti_sym_UPDOWN " << anti_sym_updownscaling.val << std::endl;
  std::cout << "anti_shear " << anti_shearing.val << std::endl;
  // TODO: figure out solution to exploding gradients when transformation matrix T has miniscule values and thus T_inv has ginormous ones
  //ADDouble9 determinant_energy = og_area*(det*det + det_inv*det_inv);
  ADDouble9 result = og_area*og_area*(anti_sym_updownscaling + anti_shearing); //og_area*(det*det + det_inv*det_inv + anti_sym_updownscaling + anti_shearing);
  //Eigen::VectorXd grad = result.grad;
  // TODO: set gradient
  //gradient = result.grad;
  std::cout << "pre gradient push" << std::endl;
  gradient << result.grad;
  std::cout << "pre result_sum" << std::endl;
  std::cout << "result_sum: " << result_sum << ", result.val: " << result.val << std::endl;
  result_sum += result.val;

  if ((select_first_point*gradient).norm() > 1.0
  || (select_second_point*gradient).norm() > 1.0
  || (select_third_point*gradient).norm() > 1.0)
  {
    std::cout << "Found exploding gradient!" << std::endl;
    //std::cout << "------ determinant energy: (" << determinant_energy.val << ")" << std::endl;
    std::cout << "------ anti_sym_updownscaling: (" << anti_sym_updownscaling.val << ")" << std::endl;
    std::cout << "------ anti_shearing: (" << anti_shearing.val << ")" << std::endl;
    std::cout << "(" << gradient(0) << ", " << gradient(1) << ", " << gradient(2) << ")" << std::endl;

    std::cout << "------ det: " << det.val << ", det_inv: " << det_inv.val << std::endl;

    std::cout << "------ og_pointp1 (2D): (" << og2d_p1[0] << ", " << og2d_p1[1] << ")" << std::endl;
    std::cout << "------ og_pointp2 (2D): (" << og2d_p2[0] << ", " << og2d_p2[1] << ")" << std::endl;
    std::cout << "------ op_pointp1 (2D): (" << op2d_p1[0].val << ", " << op2d_p1[1].val << ")" << std::endl;
    std::cout << "------ op_pointp2 (2D): (" << op2d_p2[0].val << ", " << op2d_p2[1].val << ")" << std::endl;

    std::cout << "Og transformation matrix:" << std::endl;
    for (int i=0; i<Og.rows(); i++)
    {
      for (int j=0; j<Og.cols(); j++)
      {
        std::cout << "[ " << Og(i, j) << " ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Op transformation matrix:" << std::endl;
    for (int i=0; i<saveOp.rows(); i++)
    {
      for (int j=0; j<saveOp.cols(); j++)
      {
        std::cout << "[ " << saveOp(i, j).val << " ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Og inverse transformation matrix:" << std::endl;
    for (int i=0; i<Og_inv.rows(); i++)
    {
      for (int j=0; j<Og_inv.cols(); j++)
      {
        std::cout << "[ " << Og_inv(i, j) << " ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "T transformation matrix:" << std::endl;
    for (int i=0; i<T.rows(); i++)
    {
      for (int j=0; j<T.cols(); j++)
      {
        std::cout << "[ " << T(i, j).val << " ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "T inverse transformation matrix:" << std::endl;
    for (int i=0; i<T_inv.rows(); i++)
    {
      for (int j=0; j<T_inv.cols(); j++)
      {
        std::cout << "[ " << T_inv(i, j).val << " ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

/*   std::cout << "Verts vector:" << std::endl;
  for (int i=0; i<verts.size(); i++)
  {
    std::cout << "[ " << verts(i).val << " ]";
  }
  std::cout << std::endl;

  std::cout << "select first point matrix:" << std::endl;
  for (int i=0; i<select_first_point.rows(); i++)
  {
    for (int j=0; j<select_first_point.cols(); j++)
    {
      std::cout << "[ " << select_first_point(i, j) << " ]";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "select second point matrix:" << std::endl;
  for (int i=0; i<select_second_point.rows(); i++)
  {
    for (int j=0; j<select_second_point.cols(); j++)
    {
      std::cout << "[ " << select_second_point(i, j) << " ]";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "select third point matrix:" << std::endl;
  for (int i=0; i<select_third_point.rows(); i++)
  {
    for (int j=0; j<select_third_point.cols(); j++)
    {
      std::cout << "[ " << select_third_point(i, j) << " ]";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl; */

  return std::pair<Eigen::Matrix3d, double>(gradient, result_sum);
}

std::pair<Eigen::Vector3d, double> asdap_energy_vertex_gradient(const ASDAPData& data, const Eigen::MatrixXd& U, int vertex_idx){
  std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();
  Eigen::Vector3d gradient({0, 0, 0});
  double result_sum = 0.0;

  // iterate through all incident faces
  gcs::Vertex mesh_vertex = data.inputMesh->vertex(vertex_idx);
  gc::NavigationSetBase<gcs::VertexAdjacentFaceNavigator> mesh_faces = mesh_vertex.adjacentFaces();
  for (gcs::Face mesh_face : mesh_faces)
  {
    std::vector<size_t> face = faces[mesh_face.getIndex()];
    Eigen::MatrixXd OG = data.V;
    Eigen::MatrixXd OP = U;
    Eigen::Vector3<ADDouble> vertex = ADDouble::make_active({OP(vertex_idx, 0), OP(vertex_idx, 1), OP(vertex_idx, 2)});

    Eigen::Vector3d og_pointp1;
    Eigen::Vector3d og_pointp2;
    og_pointp1 = OG.row(face[1]) - OG.row(face[0]);
    og_pointp2 = OG.row(face[2]) - OG.row(face[0]);
    Eigen::Vector3<ADDouble> op_pointp1;
    Eigen::Vector3<ADDouble> op_pointp2;
    if (face[0] == vertex_idx)
    {
      op_pointp1 = Eigen::Vector3d(OP.row(face[1])) - vertex;
      op_pointp2 = Eigen::Vector3d(OP.row(face[2])) - vertex;
    }
    else if (face[1] == vertex_idx)
    {
      op_pointp1 = vertex - Eigen::Vector3d(OP.row(face[0]));
      op_pointp2 = Eigen::Vector3d(OP.row(face[2])) - Eigen::Vector3d(OP.row(face[0]));
    }
    else
    {
      op_pointp1 = Eigen::Vector3d(OP.row(face[1])) - Eigen::Vector3d(OP.row(face[0]));
      op_pointp2 = vertex - Eigen::Vector3d(OP.row(face[0]));
    }

    double og_area = og_pointp1.cross(og_pointp2).norm();

    Eigen::Vector2d lead = Eigen::Vector2d({1, 0});
    Eigen::Vector2d trail = Eigen::Vector2d({0, 1});

    Eigen::Vector2<ADDouble> op_lead = Eigen::Vector2<ADDouble>({1, 0});
    Eigen::Vector2<ADDouble> op_trail = Eigen::Vector2<ADDouble>({0, 1});

    double og2d_p1x = og_pointp1.norm();
    double og2d_p2x = og_pointp1.dot(og_pointp2) / og2d_p1x;
    double og2d_p2y = sqrt(og_pointp2.squaredNorm() - og2d_p2x * og2d_p2x);
    Eigen::Vector2d og2d_p1 = og2d_p1x * lead;
    Eigen::Vector2d og2d_p2 = og2d_p2x * lead + og2d_p2y * trail;

    ADDouble op2d_p1x = op_pointp1.norm();
    ADDouble op2d_p2x = op_pointp1.dot(op_pointp2) / op2d_p1x;
    ADDouble op2d_p2y = sqrt(op_pointp2.squaredNorm() - op2d_p2x * op2d_p2x);
    Eigen::Vector2<ADDouble> op2d_p1 = op2d_p1x * op_lead;
    Eigen::Vector2<ADDouble> op2d_p2 = op2d_p2x * op_lead + op2d_p2y * op_trail;

    Eigen::Matrix2d Og(2, 2);
    Eigen::Matrix2<ADDouble> Op(2, 2);

    Og = og2d_p1*lead.transpose() + og2d_p2*trail.transpose();
    Op = op2d_p1*op_lead.transpose() + op2d_p2*op_trail.transpose();
    Eigen::Matrix2d Og_inv = Og.inverse();

    Eigen::Matrix2<ADDouble> saveOp = Op; // TODO: delete later ONLY USED FOR PRINTS
    Op.applyOnTheRight(Og_inv);
    Eigen::Matrix2<ADDouble> T = Op;
    Eigen::Matrix2<ADDouble> T_inv = T.inverse();

    // TODO: figure out solution to exploding gradients when transformation matrix T has miniscule values and thus T_inv has ginormous ones
    ADDouble det = T.determinant();
    ADDouble det_inv = T_inv.determinant();

    ADDouble anti_sym_updownscaling = T(0,0)*T(0,0) + T(1, 1)*T(1,1)
                                      + T_inv(0,0)*T_inv(0,0) + T_inv(1, 1)*T_inv(1,1);
    ADDouble anti_shearing = T(0,1)*T(0,1) + T(1, 0)*T(1,0)
                                      + T_inv(0,1)*T_inv(0,1) + T_inv(1, 0)*T_inv(1,0);

    ADDouble determinant_energy = og_area*(det*det + det_inv*det_inv);
    ADDouble result = /* og_area* */og_area*(anti_sym_updownscaling + anti_shearing); //og_area*(det*det + det_inv*det_inv + anti_sym_updownscaling + anti_shearing);
    Eigen::Vector3d grad = result.grad;
    gradient += grad;
    result_sum += result.val;
    //gradient += det.grad;

/*     if (grad.norm() > 1.0)
    {
      std::cout << "Found exploding gradient!" << std::endl;
      std::cout << "------ determinant energy: (" << determinant_energy.val << ")" << std::endl;
      std::cout << "------ anti_sym_updownscaling: (" << anti_sym_updownscaling.val << ")" << std::endl;
      std::cout << "------ anti_shearing: (" << anti_shearing.val << ")" << std::endl;
      std::cout << "(" << grad(0) << ", " << grad(1) << ", " << grad(2) << ")" << std::endl;

      std::cout << "------ det: " << det.val << ", det_inv: " << det_inv.val << std::endl;

      std::cout << "------ og_pointp1 (2D): (" << og2d_p1[0] << ", " << og2d_p1[1] << ")" << std::endl;
      std::cout << "------ og_pointp2 (2D): (" << og2d_p2[0] << ", " << og2d_p2[1] << ")" << std::endl;
      std::cout << "------ op_pointp1 (2D): (" << op2d_p1[0].val << ", " << op2d_p1[1].val << ")" << std::endl;
      std::cout << "------ op_pointp2 (2D): (" << op2d_p2[0].val << ", " << op2d_p2[1].val << ")" << std::endl;

      printEigenMatrix("Og transformation", Og);
      printTinyADMatrix("Op transformation", saveOp);
      printEigenMatrix("Og_inv transformation", Og_inv);
      printTinyADMatrix("T transformation", T);
      printTinyADMatrix("T_inv transformation", T_inv);
    } */
  }

  //printEigenMatrix("gradient", gradient);
  //gradient = Eigen::Vector3d({1.0 / gradient(0), 1.0 / gradient(1), 1.0 / gradient(2)});
  //printEigenMatrix("gradient", gradient);

  //std::cout << "Exiting gradient calculation for vertex " << vertex_idx << std::endl;
  return std::pair<Eigen::Vector3d, double>(gradient, result_sum);
}

// Vertex based
std::pair<Eigen::MatrixXd, double> asdap_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U){
  int UR = U.rows();
  int UC = U.cols();
  Eigen::MatrixXd gradients(UR, UC);
  double energy = 0.0;
  bool foundGradient = false;

  // create isVertexConstrained lookup table for efficiency
  std::vector<bool> isVertexConstrained = std::vector<bool>(data.V.rows());
  for (int i=0; i<data.V.rows(); i++) { isVertexConstrained[i] = false; }
  for (int i=0; i<constraints_indices.size(); i++) { isVertexConstrained[constraints_indices(i)] = true; }

  // std::cout << "Found number of vertices: " << data.V.rows() << std::endl;
  // iterate over all vertices // TODO: change to a face based approach (This one is performance unfriendly, because the energy for each face is calculated once for each vertex around that face.) Instead go through the faces and evaluate each face for each of its adjacent vertices at once. Then simply sum up the vertex gradients in a global gradient list (in the right entries).
  for (int i=0; i<data.V.rows(); i++)
  {
    //std::cout << "------ Checking vertex" << i << std::endl;
    // skip constrained vertices & non-changed vertices
    // TODO: because of changed optimisation energy becomes lower (fix by always saving the last energy value for each vertex & adding it to the energy here)
    if (isVertexConstrained[i] || !data.changedV[i]) {
      gradients(i, 0) = 0.0;
      gradients(i, 1) = 0.0;
      gradients(i, 2) = 0.0;
      //std::cout << "------> vertex" << i << " is constrained or unchanged" << std::endl;
      continue;
    }

    std::cout << "------ Calculating gradient for vertex" << i << std::endl;

    // collect the energy & gradient for each vertex
    std::pair<Eigen::Vector3d, double> result = asdap_energy_vertex_gradient(data, U, i);
    Eigen::Vector3d v_grad = result.first;
    gradients(i, 0) = v_grad(0);
    gradients(i, 1) = v_grad(1);
    gradients(i, 2) = v_grad(2);
    energy += result.second;

    // set the vertices to changed, that were affected by this vertices change
    if (v_grad.norm() > /* std::numeric_limits<double>::epsilon */ 0.00001)
    {
      foundGradient = true;
      data.nextChangedV[i] = true;
      gcs::Vertex v = data.inputMesh->vertex(i);
      for (gcs::Vertex av : v.adjacentVertices())
      {
        int av_idx = av.getIndex();
        data.nextChangedV[av_idx] = true;
      }
    }
  }

  data.hasConverged = !foundGradient;
  data.changedV = data.nextChangedV;
  for (int i=0; i<data.V.size(); i++) { data.nextChangedV[i] = false; }
  // return the aggregate of the vertex gradients
  return std::pair<Eigen::MatrixXd, double>(gradients, energy);
}
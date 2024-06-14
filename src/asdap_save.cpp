#include "asdap.hpp"

// Old & det formulation of the energy
/* double asdap_energy(const ASDAPData& data, const Eigen::MatrixXd& U){
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
    if (fabs(det - 1.0f) < std::numeric_limits<float>::epsilon() || fabs(det_inv - 1.0f) < std::numeric_limits<float>::epsilon()) {
      std::cout << "found transformed triangle with det = " + std::to_string(det) +" & det_inv = " + std::to_string(det_inv) << std::endl;
      std::cout << "Double triangle area = " + std::to_string(og_area) << std::endl;
    }
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
} */

// asdap energy gradient with prints
/*std::pair<Eigen::Vector3d, double> asdap_energy_vertex_gradient(const ASDAPData& data, const Eigen::MatrixXd& U, int vertex_idx){
  std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();
  Eigen::Vector3d gradient({0, 0, 0});
  double result_sum = 0.0;

  // iterate through all incident faces
  gcs::Vertex mesh_vertex = data.inputMesh->vertex(vertex_idx);
  gc::NavigationSetBase<gcs::VertexAdjacentFaceNavigator> mesh_faces = mesh_vertex.adjacentFaces();
  //std::cout << "Calculating the vertex v" << vertex_idx << " gradient:" << std::endl;
  int n = 0;
  for (gcs::Face mesh_face : mesh_faces)
  {
    n++;
    //if (vertex_idx == 100) std::cout << "---------------- " << "Energy gradient iteration for face: " << mesh_face << " ----------------" << std::endl;
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
    std::cout << OP.row(face[0]) << std::endl;
    std::cout << OP.row(face[1]) << std::endl;
    std::cout << OP.row(face[2]) << std::endl;
    std::cout << std::endl;
    std::cout << OP.row(face[1]) - OP.row(face[0]) << std::endl;
    std::cout << OP.row(face[2]) - OP.row(face[0]) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << vertex << std::endl;
    std::cout << "(" << vertex(0).grad << ", " << vertex(1).grad << ", " << vertex(2).grad << ")\n" << std::endl;
    std::cout << OP.row(face[1]) << std::endl;
    std::cout << OP.row(face[2]) << std::endl;
    std::cout << std::endl;
    // std::cout << "\n--- for face: " << mesh_face;
    if (face[0] == vertex_idx)
    {
      // std::cout << " (1st face vertex): ";
      op_pointp1 = Eigen::Vector3d(OP.row(face[1])) - vertex;
      op_pointp2 = Eigen::Vector3d(OP.row(face[2])) - vertex;
    }
    else if (face[1] == vertex_idx)
    {
      // std::cout << " (2nd face vertex): ";
      op_pointp1 = vertex - Eigen::Vector3d(OP.row(face[0]));
      op_pointp2 = Eigen::Vector3d(OP.row(face[2])) - Eigen::Vector3d(OP.row(face[0]));
    }
    else if (face[2] == vertex_idx)
    {
      // std::cout << " (3rd face vertex): ";
      op_pointp1 = Eigen::Vector3d(OP.row(face[1])) - Eigen::Vector3d(OP.row(face[0]));
      op_pointp2 = vertex - Eigen::Vector3d(OP.row(face[0]));
    }

    double og_area = og_pointp1.cross(og_pointp2).norm();
    //if (og_area <= 0) { std::cout << "OG_AREA = " << og_area << " <= 0" << std::endl; }

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

    Eigen::Vector3d lead = Eigen::Vector3d({1, 0, 0});
    Eigen::Vector3d center = Eigen::Vector3d({0, 1, 0});
    Eigen::Vector3d trail = Eigen::Vector3d({0, 0, 1});

    Eigen::Vector3<ADDouble> op_lead = Eigen::Vector3<ADDouble>({1, 0, 0});
    Eigen::Vector3<ADDouble> op_center = Eigen::Vector3<ADDouble>({0, 1, 0});
    Eigen::Vector3<ADDouble> op_trail = Eigen::Vector3<ADDouble>({0, 0, 1});

    // TODO: instead of constructing 3x3 matrices and determining their determinants, it might be more efficient to project the triangles into 2d space and calculate the determinants there. As we are only interested in the determinant values, we dont even care to preserve rotation or translation.
    Eigen::Vector3d og_orthogonal = og_pointp1.cross(og_pointp2).normalized();
    Eigen::Vector3<ADDouble> op_orthogonal = op_pointp1.cross(op_pointp2).normalized();

    Eigen::Matrix3d Og(3, 3);
    Eigen::Matrix3<ADDouble> Op(3, 3);
    Og = og_orthogonal*lead.transpose() + og_pointp1*center.transpose() + og_pointp2*trail.transpose();
    Op = op_orthogonal*op_lead.transpose() + op_pointp1*op_center.transpose() + op_pointp2*op_trail.transpose();

    Eigen::Matrix2d Og_inv = Og.inverse();
    // Eigen::Matrix3d Og_inv = Og.inverse();

    Eigen::MatrixX<ADDouble> saveOp = Op; // TODO: delete later ONLY USED FOR PRINTS
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
    // NOTE: maybe remove one og_area? Adding the second one was only a test to see if it would remedy some of the exploding gradients
    ADDouble result = og_area*og_area*(anti_sym_updownscaling + anti_shearing); //og_area*(det*det + det_inv*det_inv + anti_sym_updownscaling + anti_shearing);
    Eigen::Vector3d grad = result.grad;
    gradient += grad;
    result_sum += result.val;
    //gradient += det.grad;

    if (grad.norm() > 1.0)
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
    std::cout << "------ determinant energy: (" << determinant_energy.val << ")" << std::endl;
    std::cout << "------ anti_sym_updownscaling: (" << anti_sym_updownscaling.val << ")" << std::endl;
    std::cout << "------ anti_shearing: (" << anti_shearing.val << ")" << std::endl;
    std::cout << "(" << grad(0) << ", " << grad(1) << ", " << grad(2) << ")" << std::endl;

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

    if (result != result) {
      std::cout << "Face " + std::to_string(vertex_idx) + ": (original) <" + std::to_string(OG(face[0], 0)) + ", "
                                                                  + std::to_string(OG(face[0], 1)) + ", "
                                                                  + std::to_string(OG(face[0], 2)) + ">, "
                                                            + "<" + std::to_string(OG(face[1], 0)) + ", "
                                                                  + std::to_string(OG(face[1], 1)) + ", "
                                                                  + std::to_string(OG(face[1], 2)) + ">, "
                                                            + "<" + std::to_string(OG(face[2], 0)) + ", "
                                                                  + std::to_string(OG(face[2], 1)) + ", "
                                                                  + std::to_string(OG(face[2], 2)) + ">" << std::endl;
      std::cout << "Face " + std::to_string(vertex_idx) + ": (optimised) <" + std::to_string(OP(face[0], 0)) + ", "
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
    }
  }
  gradient /= n;

  //std::cout << "Exiting gradient calculation for vertex " << vertex_idx << std::endl;
  return std::pair<Eigen::Vector3d, double>(gradient, result_sum);
}*/

/* // Vertex based without active vertex optimisation
std::pair<Eigen::MatrixXd, double> asdap_energy_gradient(const Eigen::VectorXi& constraints_indices, ASDAPData& data, const Eigen::MatrixXd& U){
  int UR = U.rows();
  int UC = U.cols();
  Eigen::MatrixXd gradients(UR, UC);
  double energy = 0.0;

  // TODO: add optimisation to only iterate over "activated" vertices. Activated vertices are vertices, whose neighbours have been changed in the last iteration (either by a constraint being set, or them being updated by a gradient step)

  // create isVertexConstrained lookup table for efficiency
  std::vector<bool> isVertexConstrained = std::vector<bool>(data.V.rows());
  for (int i=0; i<data.V.rows(); i++) { isVertexConstrained[i] = false; }
  for (int i=0; i<constraints_indices.size(); i++) { isVertexConstrained[constraints_indices(i)] = true; }

  std::cout << "Found number of vertices: " << data.V.rows() << std::endl;
  // iterate over all vertices // TODO: change to a face based approach (This one is performance unfriendly, because the energy for each face is calculated once for each vertex around that face.) Instead go through the faces and evaluate each face for each of its adjacent vertices at once. Then simply sum up the vertex gradients in a global gradient list (in the right entries).
  for (int i=0; i<data.V.rows(); i++)
  {
    // skip constrained vertices & non-changed vertices
    if (isVertexConstrained[i]) {
      gradients(i, 0) = 0.0;
      gradients(i, 1) = 0.0;
      gradients(i, 2) = 0.0;
      continue;
    }

    // collect the energy & gradient for each vertex
    std::pair<Eigen::Vector3d, double> result = asdap_energy_vertex_gradient(data, U, i);
    Eigen::Vector3d v_grad = result.first;
    gradients(i, 0) = v_grad(0);
    gradients(i, 1) = v_grad(1);
    gradients(i, 2) = v_grad(2);
    energy += result.second;
  }

  // return the aggregate of the vertex gradients
  return std::pair<Eigen::MatrixXd, double>(gradients, energy);
} */

// energy gradient for face-wise traversal
/* std::pair<Eigen::MatrixXd, double> asdap_energy_gradient(const Eigen::VectorXi& constraints_indices, const ASDAPData& data, const Eigen::MatrixXd& U){
  std::vector<std::vector<size_t>> faces = data.inputMesh->getFaceVertexList();
  //Eigen::Matrix3Xi faces = data.F;
  int UR = U.rows();
  int UC = U.cols();
  Eigen::MatrixXd gradients(UR, UC);
  double energy = 0.0;

  // TODO: add optimisation to only iterate over "activated" vertices. Activated vertices are vertices, whose neighbours have been changed in the last iteration (either by a constraint being set, or them being updated by a gradient step)

  // create isVertexConstrained lookup table for efficiency
  std::vector<bool> isVertexConstrained = std::vector<bool>(data.V.rows());
  for (int i=0; i<data.V.rows(); i++) { isVertexConstrained[i] = false; }
  for (int i=0; i<constraints_indices.size(); i++) { isVertexConstrained[constraints_indices(i)] = true; }

  std::cout << "Found number of vertices: " << data.V.rows() << std::endl;

  gcs::Face mesh_face = data.inputMesh->face(0);
  // iterate over all faces
  for (gcs::Face f : data.inputMesh->faces())
  {
    int face_idx = f.getIndex();
    std::cout << "Going to calculate face" << face_idx << " gradients" << std::endl;
    std::cout << "face: " << f << std::endl;
    // collect the energy & gradient for each vertex
    std::pair<Eigen::Matrix3d, double> result = asdap_energy_face_vertex_gradients(data, U, faces[face_idx]);
    std::cout << "Calculated face" << face_idx << " gradients" << std::endl;
    for (int j=0; j<3; j++)
    {
      std::cout << "Summing gradient for vertex" << j << std::endl;
      int vertex_idx = faces[face_idx][j];
      std::cout << "Got vertex index " << vertex_idx << std::endl;
      // skip constrained vertices
      if (isVertexConstrained[vertex_idx]) {
        std::cout << "vertex is constrained. Continuing..." << std::endl;
        gradients(vertex_idx, 0) = 0.0;
        gradients(vertex_idx, 1) = 0.0;
        gradients(vertex_idx, 2) = 0.0;
        continue;
      }

      std::cout << "getting grad" << std::endl;
      // TODO: fix error
      Eigen::Vector3d v_grad = result.first.row(j);

      std::cout << "gradient: (" << v_grad(0) << ", " << v_grad(1) << ", " << v_grad(2) << ")" << std::endl;
      gradients(face_idx, 0) += v_grad(0);
      gradients(face_idx, 1) += v_grad(1);
      gradients(face_idx, 2) += v_grad(2);
    }
    energy += result.second;
  }
  std::cout << "done with gradients" << std::endl;
  // return the aggregate of the vertex gradients
  return std::pair<Eigen::MatrixXd, double>(gradients, energy);
} */
#define IGL_VIEWER_VIEWER_QUIET
// #define MIN_QUAD_WITH_FIXED_CPP_DEBUG
#include <igl/opengl/glfw/Viewer.h>
#include <igl/AABB.h>
#include <igl/screen_space_selection.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/SelectionWidget.h>
#include <igl/arap.h>
#include <igl/png/readPNG.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>

#include "asdap.hpp"
#include "screenshot.hpp"

// TODO: move this somewhere else
enum class ASDAP_MODE {
  ON_INPUT,
  EACH_FRAME
};

int main(int argc, char *argv[])
{
  if (argc<2) {
    std::cout << "Please pass an input mesh as first argument" << std::endl;
    exit(1);
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argv[1], V, F);

  Eigen::MatrixXd U = V;

  // settings
  int mode = 0;
  bool virgin = true;
  bool view_constraints = true;
  bool view_all_points = false;
  bool optimise = false;
  bool showGradients = false;

  // ASDAP (as-symmetric-dirichlet-as-possible)
  ASDAP_MODE optimMode = ASDAP_MODE::ON_INPUT;

  ASDAPData Adata;
  initMesh(V, F, Adata);
  Eigen::MatrixXd gradients;
  double lr = 0.005;
  std::pair<Eigen::MatrixXd, double> result; // result of the last optimisation step
  std::ofstream logFile; // logfile
  bool doLogging = false;

  // Viewer
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;

  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
  igl::png::readPNG(argc > 2 ? argv[2] : "../textures/texture.png", R,G,B,A);

  Eigen::MatrixXd CM = (Eigen::MatrixXd(9,3)<<
      //0.8,0.8,0.8,                        // 0 gray
      1,1,1,
      1.0/255.0,31.0/255.0,255.0/255.0,   // 1 blue
      0/255.0,201.0/255.0,195.0/255.0,    // 2 cyan
      7.0/255.0,152.0/255.0,59.0/255.0,   // 3 green
      190.0/255.0,255.0/255.0,0.0/255.0,  // 4 lime
      255.0/255.0,243.0/255.0,0.0/255.0,  // 5 yellow
      245.0/255.0,152.0/255.0,0.0/255.0,  // 6 orange
      245.0/255.0,152.0/255.0,0.0/255.0,  // 7 orange
      // 224.0/255.0,18.0/255.0,0.0/255.0,   // 7 red
      220.0/255.0,13.0/255.0,255.0/255.0  // 8 magenta
    ).finished();

  auto init_viewer_data = [&](){
    viewer.data().set_face_based(true);
    viewer.data().set_texture(R,G,B,A);
    viewer.data().show_texture = true;
    viewer.data().use_matcap = true;
    viewer.data().point_size = 15;
    viewer.data().line_width = 1;
    viewer.data().show_lines = F.rows() < 20000;
  };

  // Selection
  bool translate = false;
  RAXIS rot_axis = RAXIS::YAW;
  bool only_visible = false;
  igl::opengl::glfw::imgui::SelectionWidget selection;
  Eigen::Array<double,Eigen::Dynamic,1> W = Eigen::Array<double,Eigen::Dynamic,1>::Zero(V.rows());
  Eigen::Array<double,Eigen::Dynamic,1> and_visible = Eigen::Array<double,Eigen::Dynamic,1>::Zero(V.rows());

  Eigen::MatrixXd all_constraints = V;

  Eigen::Array<int,-1,1> constraints_mask = Eigen::Array<int,-1,1>::Zero(V.rows());
  Eigen::Array<int,-1,1> selected_mask = Eigen::Array<int,-1,1>::Zero(V.rows());
  Eigen::VectorXi selected_ids;

  Eigen::MatrixXd constraints;
  Eigen::VectorXi bi;
  Eigen::VectorXi indices = Eigen::VectorXd::LinSpaced(V.rows(), 0.0, V.rows() - 1).cast<int>();


  igl::AABB<Eigen::MatrixXd, 3> tree;
  tree.init(V,F);

  // TODO: maybe visualise overall/scaling/bending gradient as different edges with different colors
  const auto update_edges = [&]()
  {
    int UR = U.rows();
    int UC = U.cols();

    Eigen::MatrixXd grad_offset = U - gradients;
    Eigen::MatrixXi E = Eigen::MatrixXi(UR, 2);
    for (int i = 0; i<UR; i++)
    {
      E(i, 0) = i;
      E(i, 1) = UR + i;
    }
    // vertical stacking
    Eigen::MatrixXd P(U.rows()+grad_offset.rows(), U.cols());
    P << U, grad_offset;
    //viewer.data().line_width = 2.0; // NOTE: SUPPOSEDLY NOT SUPPORTED ON MAC & WINDOWS
    viewer.data().set_edges(P, E, CM.row(4));
  };

  const auto update_mesh_points = [&](Eigen::VectorXi& selected_indices, Eigen::MatrixXd& selected_constraints)
  {
    for (int i=0; i<selected_indices.size(); i++)
    {
      int index = selected_indices(i);
      float c0 = selected_constraints(i, 0);
      float c1 = selected_constraints(i, 1);
      float c2 = selected_constraints(i, 2);
      if (c0 != c0 || c1 != c1 || c2 != c2)
      {
        std::cout << "found nan value: " << constraints(index) << std::endl;
      }
      U(index, 0) = c0;
      U(index, 1) = c1;
      U(index, 2) = c2;
      //std::cout << "Set mesh point " << i << std::endl;
    }
  };

  // TODO: maybe make selected points translate with vertices during optimisation
  const auto update_points = [&]()
  {
    // constrained
    viewer.data().clear_points();
    if (view_constraints || view_all_points) {
      viewer.data().set_points(igl::slice_mask(all_constraints, (selected_mask * constraints_mask).cast<bool>(), 1), CM.row(6));
      viewer.data().add_points(igl::slice_mask(/* all_constraints */ U, (selected_mask * (1-constraints_mask)).cast<bool>(), 1), CM.row(5));
      viewer.data().add_points(igl::slice_mask(all_constraints, ((1-selected_mask) * constraints_mask).cast<bool>(), 1), CM.row(7));
    }
    if (view_all_points) {
      viewer.data().add_points(igl::slice_mask(/* all_constraints */ U, ((1-selected_mask) * (1-constraints_mask)).cast<bool>(), 1), CM.row(2));
    }
  };

  // update entire mesh
  auto refresh_mesh = [&](){
    viewer.data().clear();
    viewer.data().set_mesh(U,F);
    init_viewer_data();
    update_points();
    viewer.draw();
  };

  // update mesh vertices
  auto refresh_mesh_vertices = [&](){
    viewer.data().set_vertices(U);
    init_viewer_data();
    update_points();
    viewer.draw();
  };

  selection.callback = [&]()
  {
    screen_space_selection(/* all_constraints */ U, F, tree, viewer.core().view, viewer.core().proj, viewer.core().viewport, selection.L, W, and_visible);
    if (only_visible){
      W = (W * and_visible);
    }
    selected_mask = (W.abs()>0.5).cast<int>();
    selected_ids = igl::slice_mask(indices, selected_mask.cast<bool>(), 1);
    selection.mode=selection.OFF;
    update_points();
    viewer.draw();
  };

  auto update_constraints = [&](bool draw){
    if (virgin) return;
    bi = igl::slice_mask(indices, constraints_mask.cast<bool>(), 1);
    constraints = igl::slice(all_constraints, bi, 1);
    // does all precomputations at once
    igl::parallel_for(3, [&](const int i) {
      asdap_set_constraints(bi, constraints, Adata);
      update_mesh_points(bi, constraints);
    }, 1);
    update_points();
    if(draw) { viewer.draw(); }
  };

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viewer)->bool
  {
    if (optimMode == ASDAP_MODE::EACH_FRAME && optimise && !Adata.hasConverged)
    {
      // Optimisation
      std::cout << "optimisation iteration: " << Adata.iteration;
      result = asdap_step(bi, lr, Adata, U);
      gradients = result.first;
      std::pair<double, double> optEnergies = asdap_energies(Adata, U);
      // logging
      if (doLogging)
      {
        logFile << std::to_string(optEnergies.first + optEnergies.second) << ";" << std::to_string(optEnergies.first) << ";" << std::to_string(optEnergies.second) << "\n";
      }
      std::cout << " => Energy: " << (optEnergies.first + optEnergies.second) << " (Scaling: " << optEnergies.first << ", Rotation: " << optEnergies.second << ")" << std::endl;
      // update viewer
      viewer.data().set_vertices(U);
      update_points();
      if(showGradients) { update_edges(); }
    }
    if (optimise && Adata.hasConverged)
    {
      if (doLogging) { (logFile).close(); doLogging = false; }
      std::cout << "optimisation converged!" << std::endl;
      optimise = false;
    }
    return false;
  };

  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    translate = false;
    selection.mode=selection.OFF;
    update_constraints(true);
    refresh_mesh_vertices();
    return false;
  };

  Eigen::RowVector3f last_mouse;
  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int x, int y)
  {
    if (!translate) return false;
    Eigen::RowVector3f drag_mouse(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y, last_mouse(2));
    Eigen::RowVector3f drag_scene,last_scene;
    igl::unproject(drag_mouse, viewer.core().view, viewer.core().proj, viewer.core().viewport, drag_scene);
    igl::unproject(last_mouse, viewer.core().view, viewer.core().proj, viewer.core().viewport, last_scene);
    for (size_t i = 0; i < selected_ids.size(); i++) {
      all_constraints.row(selected_ids(i)) += (drag_scene-last_scene).cast<double>();
      Adata.changedV[selected_ids(i)] = true;
    }
    constraints = igl::slice(all_constraints, bi, 1);
    last_mouse = drag_mouse;
    update_points();
    viewer.draw();
/*     auto selected_constraints = igl::slice(all_constraints, selected_ids, 1);
    asdap_set_constraints(selected_ids, selected_constraints, Adata);
    update_mesh_points(selected_ids, selected_constraints);
    refresh_mesh(); */
/*     update_constraints();
    refresh_mesh(); */
    return false;
  };



  viewer.callback_key_pressed = [&](decltype(viewer) &,unsigned int key, int mod)
  {
    std::pair<double, double> optEnergies;
    std::pair<Eigen::MatrixXd, double> result;
    std::chrono::system_clock::time_point timestamp;
    std::time_t time;
    std::string filename;
    double theta = 0;
    double U22, U21;

    switch(key)
    {
      case ' ':
        if (virgin) return true;
        if (Adata.hasConverged) return true;
        // step manually
        if (optimMode == ASDAP_MODE::ON_INPUT)
        {
          // make gradient step
          std::cout << "optimisation iteration: " << Adata.iteration;
          result = asdap_step(bi, lr, Adata, U);
          gradients = result.first;
          refresh_mesh_vertices();
          if (showGradients) { update_edges(); }

          // Logging
          optEnergies = asdap_energies(Adata, U);
          if(doLogging)
          {
            logFile << std::to_string(optEnergies.first + optEnergies.second) << ";" << std::to_string(optEnergies.first) << ";" << std::to_string(optEnergies.second) << "\n";
          }
          std::cout << " => Energy: " << (optEnergies.first + optEnergies.second) << " (Scaling: " << optEnergies.first << ", Rotation: " << optEnergies.second << ")" << std::endl;
          if (Adata.hasConverged)
          {
            if (doLogging) { (logFile).close(); doLogging = false; }
            std::cout << "optimisation converged!" << std::endl;
          }
        } else {
          optimise = !optimise;
        }
        viewer.draw();
        return true;
      case 'H': case 'h': only_visible = !only_visible; update_points(); viewer.draw(); return true;
      case 'N': case 'n':
        showGradients = !showGradients;
        if (!showGradients)
        {
          viewer.data().clear_edges();
          return true;
        }
        result = asdap_facewise_energy_gradient(bi, Adata, U);
        gradients = result.first;
        update_edges();
        viewer.draw();
        return true;
      case 'A': case 'a':
        if (doLogging)
        {
          (logFile).close();
        }
        else
        {
          doLogging = true;
          timestamp = std::chrono::system_clock::now();
          time = std::chrono::system_clock::to_time_t(timestamp);
          filename = "../logs/EnergyCurve" + std::to_string(time) + ".csv";
          logFile.open(filename);
          logFile << std::setprecision(15);
        }
        return true;
      case 'W': case 'w':
/*         // circular deformation
        for (int i = 0; i < U.rows(); i++)
        {
          double angle = U.row(i).norm() * 2.0 * 3.14159265 / 10;
          double x = cos(angle)*U(i, 0) - sin(angle)*U(i, 1);
          double y = sin(angle)*U(i, 0) + cos(angle)*U(i, 1);
          U(i, 0) = x;
          U(i, 1) = y;
        }
        // y rotation
        for (int i = 0; i < U.rows(); i++)
        {
          double yrangle = (5.0/360.0) * 2.0 * 3.14159265;
          double x1 = cos(yrangle)*U(i, 0) + sin(yrangle)*U(i, 2);
          double z1 = - sin(yrangle)*U(i, 0) + cos(yrangle)*U(i, 2);
          U(i, 0) = x1;
          U(i, 2) = z1;
        } */

/*        // scale ASDAP down in-place
        for (int i = 0; i < U.rows(); i++)
        {
          if (i < 30)
          {
            U(i, 0) = (U(i, 0) + 10) / 2.0 - 10;
            U(i, 1) = (U(i, 1) - 2.5) / 2.0 + 2.5;
            U(i, 2) = (U(i, 2) - 0.5) / 2.0 + 0.5;
          }
          else if (i < 54)
          {
            U(i, 0) = (U(i, 0) + 6) / 2.0 - 6;
            U(i, 1) = (U(i, 1) - 2.5) / 2.0 + 2.5;
            U(i, 2) = (U(i, 2) - 0.5) / 2.0 + 0.5;
          }
          else if (i < 78)
          {
            U(i, 0) = (U(i, 0) + 2) / 2.0 - 2;
            U(i, 1) = (U(i, 1) - 2.5) / 2.0 + 2.5;
            U(i, 2) = (U(i, 2) - 0.5) / 2.0 + 0.5;
          }
          else if (i < 108)
          {
            U(i, 0) = (U(i, 0) - 2) / 2.0 + 2;
            U(i, 1) = (U(i, 1) - 2.5) / 2.0 + 2.5;
            U(i, 2) = (U(i, 2) - 0.5) / 2.0 + 0.5;
          }
          else
          {
            U(i, 0) = (U(i, 0) - 6) / 2.0 + 6;
            U(i, 1) = (U(i, 1) - 2.5) / 2.0 + 2.5;
            U(i, 2) = (U(i, 2) - 0.5) / 2.0 + 0.5;
          }
        } */

/*         theta = (5.0 / 360.0) * 2.0 * 3.14159265;
        for (int i = 0; i < U.rows(); i++)
        {
          int j = 0;
          if (i < 30)
          {
            if (i != 0 && i != 1 && i != 2 && i != 3
              && i != 4 && i != 5 && i != 6 && i != 7
              && i != 8 && i != 9 && i != 10 && i != 11
              && i != 12 && i != 13 && i != 14) { continue; }
            double u0 = U(i, 0) + 10;
            double u2 = U(i, 2) - 0.5;
            double x1 = cos(theta)*u0 + sin(theta)*u2;
            double z1 = - sin(theta)*u0 + cos(theta)*u2;
            U(i, 0) = x1 - 10;
            U(i, 2) = z1 + 0.5;
          }
          else if (i < 54)
          {
            j = i - 30;
            if (j != 0 && j != 1 && j != 2 && j != 3
              && j != 4 && j != 5 && j != 6 && j != 7
              && j != 8 && j != 9 && j != 10 && j != 11) { continue; }
            double u0 = U(i, 0) + 6;
            double u2 = U(i, 2) - 0.5;
            double x1 = cos(theta)*u0 + sin(theta)*u2;
            double z1 = - sin(theta)*u0 + cos(theta)*u2;
            U(i, 0) = x1 - 6;
            U(i, 2) = z1 + 0.5;
          }
          else if (i < 78)
          {
            j = i - 54;
            if (j != 0 && j != 1 && j != 2 && j != 3
              && j != 4 && j != 5 && j != 6 && j != 7
              && j != 8 && j != 9 && j != 10 && j != 11) { continue; }
            double u0 = U(i, 0) + 2;
            double u2 = U(i, 2) - 0.5;
            double x1 = cos(theta)*u0 + sin(theta)*u2;
            double z1 = - sin(theta)*u0 + cos(theta)*u2;
            U(i, 0) = x1 - 2;
            U(i, 2) = z1 + 0.5;
          }
          else if (i < 108)
          {
            if (i != 78 && i != 79 && i != 80 && i != 81
              && i != 82 && i != 83 && i != 84 && i != 85
              && i != 86 && i != 87 && i != 88 && i != 89
              && i != 90 && i != 91 && i != 92) { continue; }
            double u0 = U(i, 0) - 2;
            double u2 = U(i, 2) - 0.5;
            double x1 = cos(theta)*u0 + sin(theta)*u2;
            double z1 = - sin(theta)*u0 + cos(theta)*u2;
            U(i, 0) = x1 + 2;
            U(i, 2) = z1 + 0.5;
          }
          else
          {
            j = i - 108;
            if (j != 0 && j != 1 && j != 2 && j != 3
              && j != 4 && j != 5 && j != 6 && j != 7
              && j != 8 && j != 9 && j != 10 && j != 11
              && j != 12 && j != 26) { continue; }
            double u0 = U(i, 0) - 6;
            double u2 = U(i, 2) - 0.5;
            double x1 = cos(theta)*u0 + sin(theta)*u2;
            double z1 = - sin(theta)*u0 + cos(theta)*u2;
            U(i, 0) = x1 + 6;
            U(i, 2) = z1 + 0.5;
          }
        } */
/*         for (int i=0; i<U.rows(); i++)
        {
          // if (i != 916 && i != 482 && i != 481 && i != 95
          // && i != 94 && i != 671 && i != 670 && i != 32
          // && i != 31 && i != 239 && i != 238 && i != 176
          // && i != 175 && i != 752 && i != 751 && i != 5
          // && i != 4 && i != 563 && i != 562 && i != 68 && i != 67
          // && i != 644 && i != 643 && i != 41 && i != 40
          // && i != 212 && i != 211 && i != 185 && i != 184
          // && i != 761 && i != 760 && i != 2 && i != 1
          // && i != 961 && i != 962 && i != 956 && i != 957
          // && i != 966 && i != 967 && i != 968 && i != 969
          // && i != 936 && i != 937 && i != 932 && i != 933
          // && i != 944 && i != 945 && i != 941 && i != 942
          // && i != 1015 && i != 1016 && i != 1012 && i != 1013
          // && i != 1020 && i != 1021 && i != 1022 && i != 1023
          // && i != 996 && i != 997 && i != 992 && i != 993
          // && i != 1001 && i != 1002 && i != 1004 && i != 1005
          // && i != 1088)
          // { continue; }
          double yrangle = (45.0/360.0) * 2.0 * 3.14159265;
          double x1 = cos(yrangle)*U(i, 0) + sin(yrangle)*U(i, 2);
          double z1 = - sin(yrangle)*U(i, 0) + cos(yrangle)*U(i, 2);
          U(i, 0) = x1 + 0.2;
          U(i, 2) = z1 + 2.0;
          U(i, 1) = U(i, 1) * (2.0 / 3.0);
        }
        constraints_mask(916) = 1;
        constraints_mask(482) = 1;
        constraints_mask(481) = 1;
        constraints_mask(95) = 1;
        constraints_mask(94) = 1;
        constraints_mask(671) = 1;
        constraints_mask(670) = 1;
        constraints_mask(32) = 1;
        constraints_mask(31) = 1;
        constraints_mask(239) = 1;
        constraints_mask(238) = 1;
        constraints_mask(176) = 1;
        constraints_mask(175) = 1;
        constraints_mask(752) = 1;
        constraints_mask(751) = 1;
        constraints_mask(5) = 1;
        constraints_mask(4) = 1;
        constraints_mask(563) = 1;
        constraints_mask(562) = 1;
        constraints_mask(68) = 1;
        constraints_mask(67) = 1;
        constraints_mask(644) = 1;
        constraints_mask(643) = 1;
        constraints_mask(41) = 1;
        constraints_mask(40) = 1;
        constraints_mask(212) = 1;
        constraints_mask(211) = 1;
        constraints_mask(185) = 1;
        constraints_mask(184) = 1;
        constraints_mask(761) = 1;
        constraints_mask(760) = 1;
        constraints_mask(2) = 1;
        constraints_mask(1) = 1;
        constraints_mask(961) = 1;
        constraints_mask(962) = 1;
        constraints_mask(956) = 1;
        constraints_mask(957) = 1;
        constraints_mask(968) = 1;
        constraints_mask(969) = 1;
        constraints_mask(966) = 1;
        constraints_mask(967) = 1;
        constraints_mask(936) = 1;
        constraints_mask(937) = 1;
        constraints_mask(932) = 1;
        constraints_mask(933) = 1;
        constraints_mask(944) = 1;
        constraints_mask(945) = 1;
        constraints_mask(941) = 1;
        constraints_mask(942) = 1;
        constraints_mask(1015) = 1;
        constraints_mask(1016) = 1;
        constraints_mask(1012) = 1;
        constraints_mask(1013) = 1;
        constraints_mask(1022) = 1;
        constraints_mask(1023) = 1;
        constraints_mask(1020) = 1;
        constraints_mask(1021) = 1;
        constraints_mask(996) = 1;
        constraints_mask(997) = 1;
        constraints_mask(992) = 1;
        constraints_mask(993) = 1;
        constraints_mask(1004) = 1;
        constraints_mask(1005) = 1;
        constraints_mask(1001) = 1;
        constraints_mask(1002) = 1;
        constraints_mask(1088) = 1; */

        // x rotation
/*         for (int i = 0; i < U.rows(); i++)
        {
          double xrangle = (135.0/360.0) * 2.0 * 3.14159265;
          double y2 = cos(xrangle)*U(i, 1) - sin(xrangle)*U(i, 2);
          double z2 = - sin(xrangle)*U(i, 0) + cos(xrangle)*U(i, 2);
          U(i, 1) = y2;
          U(i, 2) = z2;
        } */
        // fold triangle slant
        /* U(2, 2) = 0;
        U(2, 1) = -1; */

/*         // unfold triangle slant
        U(2, 2) = 1;
        U(2, 1) = 0; */

/*         // flip angled triangle slant
        U(2, 2) = -1;
        U(2, 1) = 0; */

/*         // unfold angled triangle slant
        U(2, 2) = 0;
        U(2, 1) = -1; */

        // fold angled triangle slant
        theta = 15.0 * M_PI / 180.0;
        // U(2, 2) = 0;
        // U(2, 1) = -1;
        U22 = U(2, 2);
        U21 = U(2, 1);
        U(2, 2) = sin(theta)*U21 + cos(theta)*U22;
        U(2, 1) = cos(theta)*U21 - sin(theta)*U22;
        constraints_mask(0) = 1;
        constraints_mask(1) = 1;

/*         // hourglass
        for (int i=2; i<4; i++)
        {
          U(i, 0) = -1 * U(i, 0);
          U(i, 2) = -1 * U(i, 2);

          U(i+4, 0) = -1 * U(i+4, 0);
          U(i+4, 2) = -1 * U(i+4, 2);
        } */

        virgin = false;
        // constraints_mask(0) = 1;
        // constraints_mask(2) = 1;
        // constraints_mask(8) = 1;
        // constraints_mask(9) = 1;
        // constraints_mask(10) = 1;
        // constraints_mask(11) = 1;
        // constraints_mask(12) = 1;
        // constraints_mask(13) = 1;
        // constraints_mask(14) = 1;
        // constraints_mask(15) = 1;
        // constraints_mask(16) = 1;
        // constraints_mask(17) = 1;
        all_constraints = U;
        //update_points();
        update_constraints(true);
        refresh_mesh_vertices();
        return true;
      case 'E': case 'e':
        // fold angled triangle slant
        theta = -15.0 * M_PI / 180.0;
        U22 = U(2, 2);
        U21 = U(2, 1);
        U(2, 2) = sin(theta)*U21 + cos(theta)*U22;
        U(2, 1) = cos(theta)*U21 - sin(theta)*U22;
        constraints_mask(0) = 1;
        constraints_mask(1) = 1;

        virgin = false;
        all_constraints = U;
        update_constraints(true);
        refresh_mesh_vertices();
        return true;
      case 'J': case 'j':
        optimMode = (optimMode == ASDAP_MODE::ON_INPUT) ? ASDAP_MODE::EACH_FRAME : ASDAP_MODE::ON_INPUT;
        return true;
      case 'Y': case 'y':
        (logFile).close();
        return true;
      case 'C': case 'c':
        virgin = false;
        Adata.hasConverged = false;
        constraints_mask = ((constraints_mask+selected_mask)>0).cast<int>();
        update_constraints(true);
        refresh_mesh();
        // std::cout << constraints << "\n\n\n";
        // std::cout << bi << "\n\n\n";
        return true;
      case 'U': case 'u':
        constraints_mask = ((constraints_mask-selected_mask)==1).cast<int>();
        if (constraints_mask.sum() == 0){
          virgin = true;
          return true;
        }
        update_constraints(true);
        return true;
      case 'G': case 'g':
        if (virgin) return true;
        translate = !translate;
        {
          Eigen::MatrixXd CP;
          Eigen::Vector3d mean = constraints.colwise().mean();
          igl::project(mean, viewer.core().view, viewer.core().proj, viewer.core().viewport, CP);
          last_mouse = Eigen::RowVector3f(viewer.current_mouse_x, viewer.core().viewport(3) - viewer.current_mouse_y, CP(0,2));
          update_constraints(true);
        }
        return true;
      case 'D': case 'd':
        if (virgin) return true;
        selected_mask = (W.abs()>0.5).cast<int>();
        selected_ids = igl::slice_mask(indices, selected_mask.cast<bool>(), 1);
        U = rotate_selected(U, selected_ids, rot_axis, 10.0);
        all_constraints = U;
        update_constraints(true);
        refresh_mesh_vertices();
        break;
      case 'K': case 'k':
        selection.mode = selection.RECTANGULAR_MARQUEE;
        selection.clear();
        return true;
      case 'S': case 's':
        {
          auto t = std::time(nullptr);
          auto tm = *std::localtime(&t);
          std::ostringstream oss;
          oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
          screenshot(viewer, oss.str()+".png", 1);
        }
        return true;
      case 'T': case 't':
        if (mode == 0)
        {
          viewer.data().set_texture(R,G,B,A);
        } else {
          Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
          igl::png::readPNG(argc > 3 ? argv[3] : "../textures/texture.png", R,G,B,A);
          viewer.data().set_texture(R,G,B,A);
        }
        return true;
      case 'R': case 'r':
        // reset mesh
        // TODO: also reset the constraints
        U = V; refresh_mesh(); return true;
      case 'F': case 'f':
        // filter
        selected_mask = selected_mask*constraints_mask;
        selected_ids = igl::slice_mask(indices, selected_mask.cast<bool>(), 1);
        update_points();
        viewer.draw();
        return true;
      case 'P': case 'p':
        view_all_points = !view_all_points;
        update_points();
        viewer.draw();
        return true;
      case '.':
        view_constraints = !view_constraints;
        if (!view_constraints) { view_all_points = false; }
        update_points();
        viewer.draw();
        return true;
      case '-':
        switch(rot_axis)
        {
          default:
          case RAXIS::ROLL:
            rot_axis = RAXIS::YAW;
            break;
          case RAXIS::YAW:
            rot_axis = RAXIS::PITCH;
            break;
          case RAXIS::PITCH:
            rot_axis = RAXIS::ROLL;
            break;
        }
        return true;
      case 'X': case 'x':
        if (virgin) return true;
        // reset constrainted positions of selected
        for (size_t i = 0; i < selected_mask.size(); i++) {
          if (selected_mask(i))
            all_constraints.row(i) = V.row(i);
        }
        update_constraints(true);
        return true;
      case 'Q': case 'q':
        std::vector<Eigen::VectorXd> points;
        std::vector<std::array<int, 2>> edges;
        Eigen::MatrixXd P1, P2;
        P1.resize(0,3);
        P2.resize(0,3);
        viewer.data().set_edges(P1, P2.cast<int>(), CM.row(4));
        viewer.data().add_edges(P1, P2, CM.row(4));
        return true;
    }
    return false;
  };

  viewer.plugins.push_back(&plugin);
  plugin.widgets.push_back((igl::opengl::glfw::imgui::ImGuiWidget*) &selection);
  selection.mode = selection.OFF;

  std::cout<<R"( igl::opengl::glfw::Viewer usage:
  I,i     Toggle invert normals
  L       Toggle wireframe
  O,o     Toggle orthographic/perspective projection
  Z       Snap to canonical view
  [,]     Toggle between rotation control types (trackball, two-axis
          valuator with fixed up, 2D mode with no rotation))
  <,>     Toggle between models
  ;       Toggle vertex labels
  :       Toggle face labels/

ASDAP Usage:
  [space]  Run one ASDAP Step (hold for animation)
  J,j      Toggle single Step/continual optimization (changes [space] behaviour)
  N,n      Toggle gradient visibility
  A,a      Toggle ASDAP energy logging
  R,r      Reset mesh position

  H,h      Toggle whether to take visibility into account for selection
  F,f      Remove non constrained points from selection

  C,c      Add selection to constraints
  U,u      Remove selection from constraints

  X,x      Reset constrain position of selected vertices
  G,g      Move selection
  D,d      Rotate selection
  P,p      View all point positions
  .        View constrained point positions
  -        Toggle rotation axis (Yaw -> Pitch -> Roll)

  S,s      Make screenshot
)";

  viewer.data().clear();
  viewer.data().set_mesh(V,F);
  init_viewer_data();
  viewer.core().is_animating = true;
  viewer.core().background_color.head(3) = CM.row(0).head(3).cast<float>();
  // update_points()
  viewer.launch();
}

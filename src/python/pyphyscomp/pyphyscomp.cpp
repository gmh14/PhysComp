#include "physcomp.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>

#include <iostream>
#include <chrono>
#include <fstream>

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(pypgo, m)
{
  // m.doc() = R"pbdoc(
  //     Pybind11 example plugin
  //     -----------------------

  //     .. currentmodule:: python_example

  //     .. autosummary::
  //        :toctree: _generate

  //        add
  //        subtract
  // )pbdoc";
  using pyArrayFloat = py::array_t<float, py::array::c_style | py::array::forcecast>;
  using pyArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

  m.def("run_quasi_static_sim", [](const std::string &tetMeshFile, const std::string &fixedVtxFile, const std::string &saveFile) -> TetMesh {
    if (tetMeshFile.length() == 0ull) {
      std::cerr << "zero length tet mesh filename" << std::endl;
      return TetMesh();

    }

    if (fixedVtxFile.length() == 0ull) {
      std::cerr << "zero length fixed vertices mesh filename" << std::endl;
      return TetMesh();
    }

    int *fixedVertices;
    int nFixedVertices = pgo_load_1d_int_text(fixedVtxFile.c_str(), &nFixedVertices);

    pgoTetMeshStructHandle tetmesh = pgo_create_tetmesh_from_file(tetMeshFile.c_str());

    int n = pgo_tetmesh_get_num_vertices(tetmesh);
    int nele = pgo_tetmesh_get_num_tets(tetmesh);

    ///// 1. quasi-static simulation on initShapeMeshFile
    std::vector<double> xStaticEqRes(n * 3 + nele * 6);
    pgo_create_quastic_static_sim(tetmesh, fixedVertices.data(), (int)fixedVertices.size(), xStaticEqRes.data(), nullptr, true);

    pgoTetMeshStructHandle tetMeshStaticEqRes = pgo_tetmesh_update_vertices(tetmesh, xStaticEqRes.data());
    pgo_save_tetmesh_to_file(tetMeshStaticEqRes, saveFile.c_str());

    return TetMesh(tetMeshStaticEqRes);
  });

  m.def("inverse_plasticity_opt", [](const std::string &tetMeshFile, const std::string &fixedVtxFile, const std::string &saveFile, double stepSize = 0.5, int verbose = 3) {
    pgo_inverse_plasticity_opt(tetMeshFile.c_str(), fixedVtxFile.c_str(), saveFile.c_str(), stepSize, verbose);
  });

  m.def("stablity_preprocess", [](const TriMeshGeo &surfMesh, const std::string &surfMeshFlattenedFile) -> double {
    double minZ = pgo_stablity_preprocess(surfMesh.handle, surfMeshFlattenedFile.c_str());
    return minZ;
  });

  m.def("stability_opt", [](const std::string &tetMeshFile, const std::string &fixedVtxFile, const std::string &saveFile, int verbose = 3) {
    pgo_stability_opt(tetMeshFile.c_str(), fixedVtxFile.c_str(), saveFile.c_str(), verbose);
  });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
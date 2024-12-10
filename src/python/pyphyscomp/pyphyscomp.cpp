#include "physcomp_c.h"
#include "pypgo.h"
#include "basicIO.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>

#include <iostream>
#include <chrono>
#include <fstream>

namespace py = pybind11;

void pyphyscomp_init(py::module &m)
{
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

    std::vector<int> fixedVertices;
    int r = pgo::BasicIO::read1DText(fixedVtxFile.c_str(), std::back_inserter(fixedVertices));
    if (r != 0) {
      std::cerr << "Failed to load fixed vertices. " << std::endl;
      return TetMesh();
    }

    pgoTetMeshStructHandle tetmesh = pgo_create_tetmesh_from_file(tetMeshFile.c_str());

    int n = pgo_tetmesh_get_num_vertices(tetmesh);
    int nele = pgo_tetmesh_get_num_tets(tetmesh);

    ///// 1. quasi-static simulation on initShapeMeshFile
    std::vector<double> xStaticEqRes(n * 3 + nele * 6);
    physcomp_create_quastic_static_sim(tetmesh, fixedVertices.data(), (int)fixedVertices.size(), xStaticEqRes.data(), nullptr, true);

    pgoTetMeshStructHandle tetMeshStaticEqRes = pgo_tetmesh_update_vertices(tetmesh, xStaticEqRes.data());
    pgo_save_tetmesh_to_file(tetMeshStaticEqRes, saveFile.c_str());

    return TetMesh(tetMeshStaticEqRes);
  });

  m.def("inverse_plasticity_opt", [](const std::string &tetMeshFile, const std::string &fixedVtxFile, const std::string &saveFile, double stepSize = 0.5, int verbose = 3) {
    physcomp_inverse_plasticity_opt(tetMeshFile.c_str(), fixedVtxFile.c_str(), saveFile.c_str(), stepSize, verbose);
  });

  m.def("stablity_preprocess", [](const TriMeshGeo &surfMesh, const std::string &surfMeshFlattenedFile) -> double {
    double minZ = physcomp_stablity_preprocess(surfMesh.handle, surfMeshFlattenedFile.c_str());
    return minZ;
  });

  m.def("stability_opt", [](const std::string &tetMeshFile, const std::string &fixedVtxFile, const std::string &saveFile, int verbose = 3) {
    physcomp_stability_opt(tetMeshFile.c_str(), fixedVtxFile.c_str(), saveFile.c_str(), verbose);
  });
}
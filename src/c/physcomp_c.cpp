#include "physcomp_c.h"

#include "EigenMKLPardisoSupport.h"
#include "tetMeshGeo.h"
#include "triMeshGeo.h"
#include "generateTetMeshMatrix.h"
#include "tetMesh.h"
#include "pgoLogging.h"
#include "geometryQuery.h"
#include "boundingVolumeTree.h"
#include "initPredicates.h"
#include "EigenSupport.h"
#include "simulationMesh.h"
#include "deformationModelManager.h"
#include "tetMeshDeformationModel.h"
#include "basicIO.h"
#include "deformationModelAssemblerElastic.h"
#include "deformationModelEnergy.h"
#include "plasticModel.h"
#include "multiVertexPullingSoftConstraints.h"
#include "implicitBackwardEulerTimeIntegrator.h"
#include "TRBDF2TimeIntegrator.h"
#include "generateMassMatrix.h"
#include "generateSurfaceMesh.h"
#include "barycentricCoordinates.h"
#include "triangleMeshExternalContactHandler.h"
#include "configFileJSON.h"
#include "pointPenetrationEnergy.h"
#include "triangleMeshSelfContactHandler.h"
#include "pointTrianglePairCouplingEnergyWithCollision.h"
#include "linearPotentialEnergy.h"
#include "NewtonRaphsonSolver.h"

#include "deformationModelAssemblerElasticFullPlastic.h"
#include "quadraticPotentialEnergy.h"
#include "centerOfMassMatchingEnergy.h"
#include "cgalInterface.h"

#include <tbb/enumerable_thread_specific.h>

struct pgoQuasiStaticSimStruct
{
  std::shared_ptr<pgo::SolidDeformationModel::SimulationMesh> simMesh;
  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelManager> dmm;

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElasticFullPlastic> assembler;
  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElastic> assemblerPosOnly;

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelEnergy> elasticEnergy;
  std::shared_ptr<pgo::ConstraintPotentialEnergies::MultipleVertexPulling> pullingEnergy;
  std::shared_ptr<pgo::PredefinedPotentialEnergies::LinearPotentialEnergy> externalForcesEnergy;

  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAll;
  pgo::EigenSupport::SpMatD hess;

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelEnergy> elasticEnergyPosOnly;
  std::shared_ptr<pgo::ConstraintPotentialEnergies::MultipleVertexPulling> pullingEnergyPosOnly;

  std::shared_ptr<pgo::PredefinedPotentialEnergies::QuadraticPotentialEnergy> smoothnessEnergyPlacityOnly;

  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAllPosOnly;
  pgo::EigenSupport::SpMatD hessPosOnly;

  std::shared_ptr<pgo::NonlinearOptimization::NewtonRaphsonSolver> solverPosOnly;
  std::shared_ptr<pgo::EigenSupport::EigenMKLPardisoSupport> linearSolver;

  pgo::EigenSupport::VXd fext;
};

//// NeurIPS
void physcomp_create_quastic_static_sim(pgoTetMeshStructHandle tetmeshHandle, int *fixedVerticesIDs, int numFixedVertices, double *xOpt, double *plasticityParam, bool with_gravity)
{
  namespace ES = pgo::EigenSupport;

  pgo::VolumetricMeshes::TetMesh *tetMesh = reinterpret_cast<pgo::VolumetricMeshes::TetMesh *>(tetmeshHandle);

  std::shared_ptr<pgo::SolidDeformationModel::SimulationMesh> simMesh(pgo::SolidDeformationModel::loadTetMesh(tetMesh));
  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelManager> dmm = std::make_shared<pgo::SolidDeformationModel::DeformationModelManager>();

  dmm->setMesh(simMesh.get(), nullptr, nullptr);
  dmm->init(pgo::SolidDeformationModel::DeformationModelPlasticMaterial::VOLUMETRIC_DOF6,
    pgo::SolidDeformationModel::DeformationModelElasticMaterial::STABLE_NEO, nullptr);

  std::vector<double> elementWeights(simMesh->getNumElements(), 1.0);
  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElastic> assembler =
    std::make_shared<pgo::SolidDeformationModel::DeformationModelAssemblerElastic>(dmm.get(), nullptr, 0, elementWeights.data());

  int n = simMesh->getNumVertices();
  int n3 = n * 3;
  int nele = simMesh->getNumElements();
  ES::VXd plasticity(nele * 6);

  if (plasticityParam != nullptr) {
    for (int i = 0; i < nele * 6; i++) {
      plasticity[i] = plasticityParam[i];
    }
  }
  else {
    ES::M3d I = ES::M3d::Identity();
    for (int ei = 0; ei < nele; ei++) {
      dmm->getDeformationModel(ei)->getPlasticModel()->toParam(I.data(), plasticity.data() + ei * dmm->getNumPlasticParameters());
    }
  }
  assembler->enableFullspaceScale(1);
  assembler->setFixedParameters(plasticity.data(), (int)plasticity.size());

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelEnergy> elasticEnergy = std::make_shared<pgo::SolidDeformationModel::DeformationModelEnergy>(assembler, nullptr, 0);

  ES::VXd restPosition(n3);
  for (int vi = 0; vi < n; vi++) {
    double p[3];
    simMesh->getVertex(vi, p);
    restPosition.segment<3>(vi * 3) = ES::V3d(p[0], p[1], p[2]);
  }

  // attachments
  std::vector<int> fixedVertices(numFixedVertices);
  for (int i = 0; i < numFixedVertices; i++) {
    fixedVertices[i] = fixedVerticesIDs[i];
  }
  std::sort(fixedVertices.begin(), fixedVertices.end());

  ES::VXd tgtVertexPositions(fixedVertices.size() * 3);
  for (int vi = 0; vi < (int)fixedVertices.size(); vi++) {
    tgtVertexPositions.segment<3>(vi * 3) = restPosition.segment<3>(fixedVertices[vi] * 3);
  }

  ES::SpMatD K;
  elasticEnergy->createHessian(K);
  elasticEnergy->hessian(restPosition, K);

  std::shared_ptr<pgo::ConstraintPotentialEnergies::MultipleVertexPulling> pullingEnergy = std::make_shared<pgo::ConstraintPotentialEnergies::MultipleVertexPulling>(K, restPosition.data(),
    (int)fixedVertices.size(), fixedVertices.data(), tgtVertexPositions.data(), nullptr, 0);

  ES::VXd vertexMasses(n);
  vertexMasses.setZero();
  for (int ele = 0; ele < nele; ele++) {
    double mass = tetMesh->getElementVolume(ele) * 1000.0;
    for (int j = 0; j < 4; j++) {
      vertexMasses[simMesh->getVertexIndex(ele, j)] += mass * 0.25;
    }
  }

  // pgo::NonlinearOptimization::PotentialEnergies *energyAll = new pgo::NonlinearOptimization::PotentialEnergies(n3);
  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAll = std::make_shared<pgo::NonlinearOptimization::PotentialEnergies>(n3);
  energyAll->addPotentialEnergy(elasticEnergy);
  energyAll->addPotentialEnergy(pullingEnergy, 1e4);

  ES::VXd fext(n3);
  if (with_gravity) {
    // gravity
    for (int vi = 0; vi < n; vi++) {
      fext.segment<3>(vi * 3) = ES::V3d(0, 0, 9.8) * vertexMasses[vi];
    }
    std::shared_ptr<pgo::PredefinedPotentialEnergies::LinearPotentialEnergy> externalForcesEnergy = std::make_shared<pgo::PredefinedPotentialEnergies::LinearPotentialEnergy>(fext);
    energyAll->addPotentialEnergy(externalForcesEnergy);
  }

  energyAll->init();

  pgo::NonlinearOptimization::NewtonRaphsonSolver::SolverParam solverParam;

  ES::VXd x = restPosition;

  pgo::NonlinearOptimization::NewtonRaphsonSolver solver(x.data(), solverParam, energyAll, std::vector<int>(), nullptr);
  solver.solve(x.data(), 200, 1e-6, 3);

  for (int vi = 0; vi < tetMesh->getNumVertices(); vi++) {
    for (int i = 0; i < 3; i++) {
      xOpt[vi * 3 + i] = x[vi * 3 + i];
    }
  }

  // print energy
  std::cout << "Final energy: " << std::endl;
  energyAll->printEnergy(x);
}

pgoQuasiStaticSimStructHandle physcomp_create_quastic_static_sim_create_energies(pgoTetMeshStructHandle tetmeshRestHandle, pgoTetMeshStructHandle tetmeshInitHandle, int *fixedVerticesIDs, int numFixedVertices, double *xOpt, bool withGravity)
{
  namespace ES = pgo::EigenSupport;

  pgo::VolumetricMeshes::TetMesh *tetMesh = reinterpret_cast<pgo::VolumetricMeshes::TetMesh *>(tetmeshRestHandle);

  std::shared_ptr<pgo::SolidDeformationModel::SimulationMesh> simMesh(pgo::SolidDeformationModel::loadTetMesh(tetMesh));

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelManager> dmm = std::make_shared<pgo::SolidDeformationModel::DeformationModelManager>();

  dmm->setMesh(simMesh.get(), nullptr, nullptr);
  dmm->init(pgo::SolidDeformationModel::DeformationModelPlasticMaterial::VOLUMETRIC_DOF6,
    pgo::SolidDeformationModel::DeformationModelElasticMaterial::STABLE_NEO, nullptr);

  std::vector<double> elementWeights(simMesh->getNumElements(), 1.0);
  // std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElastic> assembler =
  //   std::make_shared<pgo::SolidDeformationModel::DeformationModelAssemblerElastic>(dmm.get(), nullptr, 0, elementWeights.data());
  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElasticFullPlastic> assembler =
    std::make_shared<pgo::SolidDeformationModel::DeformationModelAssemblerElasticFullPlastic>(dmm.get(),
      nullptr, 0, elementWeights.data());

  int n = simMesh->getNumVertices();
  int n3 = n * 3;
  int nele = simMesh->getNumElements();
  ES::VXd plasticity(nele * 6);
  ES::M3d I = ES::M3d::Identity();
  for (int ei = 0; ei < nele; ei++) {
    dmm->getDeformationModel(ei)->getPlasticModel()->toParam(I.data(), plasticity.data() + ei * dmm->getNumPlasticParameters());
  }
  assembler->enableFullspaceScale(1);
  // plasticity as variables
  // assembler->setFixedParameters(plasticity.data(), (int)plasticity.size());

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelEnergy> elasticEnergy = std::make_shared<pgo::SolidDeformationModel::DeformationModelEnergy>(assembler, nullptr, 0);

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElastic> assemblerPosOnly =
    std::make_shared<pgo::SolidDeformationModel::DeformationModelAssemblerElastic>(dmm.get(),
      nullptr, 0, elementWeights.data());
  assemblerPosOnly->enableFullspaceScale(1);
  assemblerPosOnly->setFixedParameters(plasticity.data(), (int)plasticity.size());

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelEnergy> elasticEnergyPosOnly = std::make_shared<pgo::SolidDeformationModel::DeformationModelEnergy>(assemblerPosOnly, nullptr, 0);

  // pgoTetMeshStructHandle tetmeshInitHandle = pgo_create_tetmesh_from_file("/home/mhg/Projects/libpgo/mesh_data/m.2.veg");
  // pgoTetMeshStructHandle tetmeshInitHandle = pgo_create_tetmesh_from_file("/home/mhg/Projects/libpgo/mesh_data/m.2.deform1.veg");
  pgo::VolumetricMeshes::TetMesh *tetmeshInit = reinterpret_cast<pgo::VolumetricMeshes::TetMesh *>(tetmeshInitHandle);
  std::shared_ptr<pgo::SolidDeformationModel::SimulationMesh> simMeshInit(pgo::SolidDeformationModel::loadTetMesh(tetmeshInit));

  ES::VXd initPos(n3);
  for (int vi = 0; vi < n; vi++) {
    double p[3];
    simMeshInit->getVertex(vi, p);
    // simMesh->getVertex(vi, p);
    initPos.segment<3>(vi * 3) = ES::V3d(p[0], p[1], p[2]);
  }

  // attachments
  std::vector<int> fixedVertices(numFixedVertices);
  for (int i = 0; i < numFixedVertices; i++) {
    fixedVertices[i] = fixedVerticesIDs[i];
  }
  std::sort(fixedVertices.begin(), fixedVertices.end());

  ES::VXd tgtVertexPositions(fixedVertices.size() * 3);
  for (int vi = 0; vi < (int)fixedVertices.size(); vi++) {
    tgtVertexPositions.segment<3>(vi * 3) = initPos.segment<3>(fixedVertices[vi] * 3);
  }

  ES::VXd initPosPlasticity(n3 + nele * 6);
  initPosPlasticity.segment(0, n3) = initPos;
  initPosPlasticity.segment(n3, nele * 6) = plasticity;

  ES::SpMatD K;
  elasticEnergy->createHessian(K);
  // elasticEnergy->hessian(initPos, K);
  elasticEnergy->hessian(initPosPlasticity, K);

  ES::SpMatD KPosOnly;
  elasticEnergyPosOnly->createHessian(KPosOnly);
  elasticEnergyPosOnly->hessian(initPos, KPosOnly);

  std::shared_ptr<pgo::ConstraintPotentialEnergies::MultipleVertexPulling> pullingEnergy = std::make_shared<pgo::ConstraintPotentialEnergies::MultipleVertexPulling>(K, initPos.data(),
    (int)fixedVertices.size(), fixedVertices.data(), tgtVertexPositions.data(), nullptr, 0);

  std::shared_ptr<pgo::ConstraintPotentialEnergies::MultipleVertexPulling> pullingEnergyPosOnly = std::make_shared<pgo::ConstraintPotentialEnergies::MultipleVertexPulling>(KPosOnly, initPos.data(),
    (int)fixedVertices.size(), fixedVertices.data(), tgtVertexPositions.data(), nullptr, 0);

  ES::VXd vertexMasses(n);
  vertexMasses.setZero();
  for (int ele = 0; ele < nele; ele++) {
    double mass = tetMesh->getElementVolume(ele) * 1000.0;
    for (int j = 0; j < 4; j++) {
      vertexMasses[simMesh->getVertexIndex(ele, j)] += mass * 0.25;
    }
  }

  pgoQuasiStaticSimStruct *simStruct = new pgoQuasiStaticSimStruct;

  // gravity
  simStruct->fext.setZero(n3);
  for (int vi = 0; vi < n; vi++) {
    fext.segment<3>(vi * 3) = ES::V3d(0, 0, 9.8) * vertexMasses[vi];
  }

  std::shared_ptr<pgo::PredefinedPotentialEnergies::LinearPotentialEnergy> externalForcesEnergy = std::make_shared<pgo::PredefinedPotentialEnergies::LinearPotentialEnergy>(simStruct->fext);

  // plasicity smoothness
  ES::SpMatD L;
  pgo::Mesh::TetMeshGeo tetMeshGeo;
  tetMesh->exportMeshGeometry(tetMeshGeo);
  pgo::SolidDeformationModel::TetMeshMatrix::generateBasicElementLaplacianMatrix(tetMeshGeo, L, 1, 1);
  ES::SpMatD L6;
  ES::expandN(L, L6, 6);
  ES::VXd L6x(nele * 6);
  ES::mv(L6, plasticity, L6x);
  L6x *= -1;
  std::shared_ptr<pgo::PredefinedPotentialEnergies::QuadraticPotentialEnergy> smoothnessEnergyPlacityOnly = std::make_shared<pgo::PredefinedPotentialEnergies::QuadraticPotentialEnergy>(L6, L6x, 1);

  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAll = std::make_shared<pgo::NonlinearOptimization::PotentialEnergies>(n3 + nele * 6);
  energyAll->addPotentialEnergy(elasticEnergy);
  energyAll->addPotentialEnergy(pullingEnergy, 1e4);

  if (withGravity) {
    std::cout << "with gravity" << std::endl;
    energyAll->addPotentialEnergy(externalForcesEnergy, 1);
  }
  else {
    std::cout << "without gravity" << std::endl;
  }

  energyAll->init();

  ES::SpMatD hess;
  energyAll->createHessian(hess);

  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAllPosOnly = std::make_shared<pgo::NonlinearOptimization::PotentialEnergies>(n3);
  energyAllPosOnly->addPotentialEnergy(elasticEnergyPosOnly);
  energyAllPosOnly->addPotentialEnergy(pullingEnergyPosOnly, 1e4);
  energyAllPosOnly->addPotentialEnergy(externalForcesEnergy, 1);
  energyAllPosOnly->init();

  ES::SpMatD hessPosOnly;
  energyAllPosOnly->createHessian(hessPosOnly);

  ///
  ES::VXd x = initPosPlasticity;

  std::cout << "initial energies" << std::endl;
  energyAll->printEnergy(x);

  pgo::NonlinearOptimization::NewtonRaphsonSolver::SolverParam solverParam;
  std::shared_ptr<pgo::NonlinearOptimization::NewtonRaphsonSolver> solverPosOnly = std::make_shared<pgo::NonlinearOptimization::NewtonRaphsonSolver>(x.data(), solverParam, energyAllPosOnly, std::vector<int>(), nullptr);

  ES::SpMatD df_dx = hess.block(0, 0, n * 3, n * 3);
  std::shared_ptr<ES::EigenMKLPardisoSupport> solver = std::make_shared<ES::EigenMKLPardisoSupport>(df_dx, ES::EigenMKLPardisoSupport::MatrixType::REAL_SYM_INDEFINITE,
    ES::EigenMKLPardisoSupport::ReorderingType::NESTED_DISSECTION, 0, 0, 1, 0, 0, 0);
  solver->analyze(df_dx);

  ///
  simStruct->simMesh = simMesh;
  simStruct->dmm = dmm;

  simStruct->assembler = assembler;
  simStruct->assemblerPosOnly = assemblerPosOnly;

  simStruct->elasticEnergy = elasticEnergy;
  simStruct->pullingEnergy = pullingEnergy;
  simStruct->externalForcesEnergy = externalForcesEnergy;

  simStruct->energyAll = energyAll;
  simStruct->hess = hess;

  simStruct->elasticEnergyPosOnly = elasticEnergyPosOnly;
  simStruct->pullingEnergyPosOnly = pullingEnergyPosOnly;

  simStruct->smoothnessEnergyPlacityOnly = smoothnessEnergyPlacityOnly;

  simStruct->energyAllPosOnly = energyAllPosOnly;
  simStruct->hessPosOnly = hessPosOnly;

  simStruct->solverPosOnly = solverPosOnly;

  simStruct->linearSolver = solver;
  for (int i = 0; i < n3 + nele * 6; i++) {
    xOpt[i] = x[i];
  }

  return reinterpret_cast<pgoQuasiStaticSimStructHandle>(simStruct);
}

void physcomp_quastic_static_gauss_newton_gradient(pgoQuasiStaticSimStructHandle quasiStaticStruct, double *dR_dx_data, double *var, int nVtx, int nElem, double *GNGrad, bool hasSmoothness)
{
  namespace ES = pgo::EigenSupport;

  std::shared_ptr<pgo::NonlinearOptimization::PotentialEnergies> energyAll = quasiStaticStruct->energyAll;

  ES::SpMatD hess = quasiStaticStruct->hess;

  if (nVtx * 3 + nElem * 6 != energyAll->getNumDOFs()) {
    std::cerr << "DOF mismatch" << std::endl;
    exit(1);
  }

  ES::VXd varVec = ES::Mp<ES::VXd>(var, energyAll->getNumDOFs());

  energyAll->printEnergy(varVec);

  memset(hess.valuePtr(), 0, sizeof(double) * hess.nonZeros());
  energyAll->hessian(varVec, hess);

  ES::SpMatD df_dx = hess.block(0, 0, nVtx * 3, nVtx * 3);
  ES::SpMatD df_dS = hess.block(0, nVtx * 3, nVtx * 3, nElem * 6);

  ES::VXd dR_dx = ES::VXd::Map(dR_dx_data, nVtx * 3);

  // df_dx * delta = dR_dx
  ES::VXd delta(nVtx * 3);

  std::shared_ptr<ES::EigenMKLPardisoSupport> solver = quasiStaticStruct->linearSolver;
  solver->factorize(df_dx);
  solver->solve(df_dx, delta.data(), dR_dx.data(), 1);

  ES::VXd GNGradVec(nElem * 6);
  ES::mv(df_dS, -delta, GNGradVec, 1);

  // ES::VXd GNGradVecDebug(nElem * 6);
  // GNGradVecDebug = df_dS.transpose() * -delta;

  if (hasSmoothness) {
    ES::VXd placity = varVec.segment(nVtx * 3, nElem * 6);
    ES::VXd gradSmoothness(nElem * 6);
    double smoothnessEngValue = quasiStaticStruct->smoothnessEnergyPlacityOnly->func(placity);
    std::cout << "smoothness energy: " << smoothnessEngValue << std::endl;
    quasiStaticStruct->smoothnessEnergyPlacityOnly->gradient(placity, gradSmoothness);
    GNGradVec += gradSmoothness * 1e-3;
  }

  ES::Mp<ES::VXd>(GNGrad, nElem * 6) = GNGradVec;
}

void physcomp_project_plastic_param(pgoQuasiStaticSimStructHandle quasiStaticStruct, int nVtx, int nElem, double *xOpt, double zeroThreshold)
{
  namespace ES = pgo::EigenSupport;

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelManager> dmm = quasiStaticStruct->dmm;
  for (int ei = 0; ei < nElem; ei++) {
    std::vector<double> param(6);
    for (int i = 0; i < 6; i++) {
      param[i] = xOpt[nVtx * 3 + ei * 6 + i];
    }

    dmm->getDeformationModel(ei)->getPlasticModel()->projectParam(param.data(), zeroThreshold);

    for (int i = 0; i < 6; i++) {
      xOpt[nVtx * 3 + ei * 6 + i] = param[i];
    }
  }
}

void physcomp_static_eq_step(pgoQuasiStaticSimStructHandle quasiStaticStruct, double *var, int nVtx, int nElem, double *varOut, int verbose)
{
  namespace ES = pgo::EigenSupport;

  std::shared_ptr<pgo::NonlinearOptimization::NewtonRaphsonSolver> solver_ptr = quasiStaticStruct->solverPosOnly;

  std::shared_ptr<pgo::SolidDeformationModel::DeformationModelAssemblerElastic> assemblerPosOnly = quasiStaticStruct->assemblerPosOnly;

  assemblerPosOnly->setFixedParameters(var + nVtx * 3, nElem * 6);

  solver_ptr->solve(var, 800, 1e-6, verbose);

  for (int i = 0; i < nVtx * 3; i++) {
    varOut[i] = var[i];
  }
}

pgoCenterOfMassEnergyStructHandle physcomp_create_center_of_mass_energy(pgoTetMeshStructHandle tetmeshHandle)
{
  namespace ES = pgo::EigenSupport;
  pgo::VolumetricMeshes::TetMesh *tetMesh = reinterpret_cast<pgo::VolumetricMeshes::TetMesh *>(tetmeshHandle);

  pgo::Mesh::TetMeshGeo tetMeshGeo;
  tetMesh->exportMeshGeometry(tetMeshGeo);

  int n = tetMeshGeo.numVertices();

  pgo::Vec3d com(0.0, 0.0, 0.0);
  double totalVolume = 0.0;
  for (int elem = 0; elem < tetMeshGeo.numTets(); elem++) {
    pgo::Vec3d a = tetMeshGeo.pos(tetMeshGeo.tetVtxID(elem, 0));
    pgo::Vec3d b = tetMeshGeo.pos(tetMeshGeo.tetVtxID(elem, 1));
    pgo::Vec3d c = tetMeshGeo.pos(tetMeshGeo.tetVtxID(elem, 2));
    pgo::Vec3d d = tetMeshGeo.pos(tetMeshGeo.tetVtxID(elem, 3));
    double tetVolume = tetMesh->getTetVolume(a, b, c, d);
    pgo::Vec3d tetCenter = (a + b + c + d) / 4.0;
    com += tetVolume * tetCenter;
    totalVolume += tetVolume;
  }
  com /= totalVolume;

  // convex hull of the base
  double minZ = 1e10;
  for (int vi = 0; vi < n; vi++) {
    minZ = std::min(minZ, tetMeshGeo.pos(vi)[2]);
  }

  std::vector<pgo::Mesh::TriMeshGeo> basePC(1);
  for (int vi = 0; vi < n; vi++) {
    if (tetMeshGeo.pos(vi)[2] < minZ + 1e-3) {
      pgo::Vec3d pos = tetMeshGeo.pos(vi);
      basePC[0].addPos(pos);
    }
  }

  pgo::Mesh::TriMeshGeo basePolygon;
  pgo::CGALInterface::convexHullMesh(basePC, basePolygon);

  // closest point on the base polygon to com
  pgo::Mesh::TriMeshBVTree bvTree;
  bvTree.buildByInertiaPartition(basePolygon);
  pgo::Mesh::TriMeshPseudoNormal pseudoNormal;
  pseudoNormal.buildPseudoNormals(basePolygon);

  auto ret = bvTree.closestTriangleQuery(basePolygon, com);
  pgo::Vec3d normal = pseudoNormal.getPseudoNormal(basePolygon.triangles().data(), ret.triID, ret.feature);

  pgo::Vec3d closestPoint = ret.closestPosition;
  pgo::Vec3d diff = closestPoint - com;
  double dotVal = diff.dot(normal);

  ES::V3d tgtCoMVec(com[0], com[1], com[2]);
  // tgtCoM will be the closest point on the base polygon with a delta
  if (dotVal <= 0)  // com is outside the base polygon
  {
    // pgo::Vec3d tgtCoMpgo::Vec3d = closestPoint - 0.01 * normal;
    // tgtCoMVec = ES::V3d(tgtCoMpgo::Vec3d[0], tgtCoMpgo::Vec3d[1], tgtCoMpgo::Vec3d[2]);
    // center of basePolygon
    pgo::Vec3d center(0.0, 0.0, 0.0);
    for (int vi = 0; vi < basePolygon.numVertices(); vi++) {
      center += basePolygon.pos(vi);
    }
    center /= basePolygon.numVertices();
    tgtCoMVec = ES::V3d(center[0], center[1], center[2]);
  }

  ES::M3d projMat = ES::M3d::Identity();
  projMat(2, 2) = 0.0;

  pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy *comEnergy = new pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy(tetMeshGeo, tgtCoMVec, projMat);

  return reinterpret_cast<pgoCenterOfMassEnergyStructHandle>(comEnergy);
}

void physcomp_center_of_mass_energy_get_tgtCoM_projMat(pgoCenterOfMassEnergyStructHandle comEnergyHandle, double *tgtCoM, double *projMat)
{
  namespace ES = pgo::EigenSupport;

  pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy *comEnergy = reinterpret_cast<pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy *>(comEnergyHandle);

  ES::Mp<ES::V3d>(tgtCoM, 3) = comEnergy->getTgtCoM();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      projMat[i * 3 + j] = comEnergy->getProjMat()(i, j);
    }
  }
}

double physcomp_center_of_mass_energy_gradient(pgoCenterOfMassEnergyStructHandle comEnergyHandle, const double *x, double *grad, double *com, int numVtx)
{
  namespace ES = pgo::EigenSupport;

  pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy *comEnergy = reinterpret_cast<pgo::PredefinedPotentialEnergies::CenterOfMassMatchingEnergy *>(comEnergyHandle);

  ES::VXd xVec(numVtx * 3);
  for (int i = 0; i < numVtx * 3; i++) {
    xVec[i] = x[i];
  }

  ES::VXd gradVec(numVtx * 3);

  double energy;
  ES::V3d comVec;
  comEnergy->compute_com_and_energy_and_grad(xVec, comVec, energy, gradVec);

  ES::Mp<ES::VXd>(grad, numVtx * 3) = gradVec;
  ES::Mp<ES::V3d>(com, 3) = comVec;
  return energy;
}

void physcomp_inverse_plasticity_opt(const char *tetMeshFile, const char *fixedVtxFile, const char *saveFolder, double stepSize, int verbose)
{
  pgoTetMeshStructHandle tetmeshRest = pgo_create_tetmesh_from_file(tetMeshFile);
  pgoTetMeshStructHandle tetmeshInit = pgo_create_tetmesh_from_file(tetMeshFile);

  std::vector<int> fixedVertices;

  std::ifstream ifs(fixedVtxFile);
  int v;
  while (ifs >> v) {
    fixedVertices.push_back(v);
  }

  pgoTetMeshStructHandle tetmeshTarget = pgo_create_tetmesh_from_file(tetMeshFile);

  int n = pgo_tetmesh_get_num_vertices(tetmeshTarget);
  int nele = pgo_tetmesh_get_num_tets(tetmeshTarget);
  std::vector<double> xOpt(n * 3 + nele * 6);

  pgoQuasiStaticSimStructHandle quasiStaticStruct = physcomp_create_quastic_static_sim_create_energies(tetmeshRest, tetmeshInit, fixedVertices.data(), (int)fixedVertices.size(), xOpt.data());

  std::vector<double> xHat(n * 3);
  pgo_tetmesh_get_vertices(tetmeshTarget, xHat.data());

  // R(x) = ||x - x_hat||^2 * 0.5
  // the mass center should be lying inside the support polygon
  // R(x) = ||x - x_hat||^2 * 0.5 +
  // dR_dx = (x - x_hat)
  std::vector<double> dR_dx(n * 3);
  // double stepSize = 0.5;

  std::cout << "stepSize: " << stepSize << "\n";

  std::vector<double> finalVtx(n * 3);
  std::vector<double> finalPlasticity(nele * 6);
  std::vector<double> xLastIter = xOpt;

  for (int iter = 0; iter < 2000; iter++) {
    for (int i = 0; i < n * 3; i++) {
      dR_dx[i] = (xOpt[i] - xHat[i]);
    }

    std::vector<double> GNGrad(nele * 6);
    physcomp_quastic_static_gauss_newton_gradient(quasiStaticStruct, dR_dx.data(), xOpt.data(), n, nele, GNGrad.data());

    for (int i = 0; i < nele * 6; i++) {
      xOpt[n * 3 + i] -= stepSize * GNGrad[i];
    }

    physcomp_project_plastic_param(quasiStaticStruct, n, nele, xOpt.data(), 1e-5);

    //
    physcomp_static_eq_step(quasiStaticStruct, xOpt.data(), n, nele, xOpt.data(), verbose);

    double eng = 0.0;
    for (int i = 0; i < n * 3; i++) {
      eng += dR_dx[i] * dR_dx[i] * 0.5;
    }

    std::cout << "======iter: " << iter << " eng: " << eng << " eng / n: " << eng / n << std::endl;

    // break condition
    if (iter > 0) {
      double diff_max = 0.0;
      for (int i = 0; i < n * 3; i++) {
        diff_max = std::max(diff_max, std::abs(xOpt[i] - xLastIter[i]));
        xLastIter[i] = xOpt[i];
      }

      if (diff_max < 1e-7) {
        break;
      }
      // if (eng / n < 5e-8) {
      //   break;
      // }
    }
  }

  for (int i = 0; i < n * 3; i++) {
    finalVtx[i] = xOpt[i];
  }

  for (int i = 0; i < nele * 6; i++) {
    finalPlasticity[i] = xOpt[n * 3 + i];
  }

  std::string saveFolderStr(saveFolder);
  pgoTetMeshStructHandle tetMeshOpt = pgo_tetmesh_update_vertices(tetmeshInit, finalVtx.data());
  pgo_save_tetmesh_to_file(tetMeshOpt, (saveFolderStr + "/opt.veg").c_str());

  std::vector<double> xRest(n * 3 + nele * 6);
  // use tetmesh as initial mesh
  physcomp_create_quastic_static_sim(tetmeshInit, fixedVertices.data(), (int)fixedVertices.size(), xRest.data(), finalPlasticity.data(), false);

  pgoTetMeshStructHandle tetMeshRestOpt = pgo_tetmesh_update_vertices(tetMeshOpt, xRest.data());
  pgo_save_tetmesh_to_file(tetMeshRestOpt, (saveFolderStr + "/opt_rest.veg").c_str());
}

double physcomp_stablity_preprocess(pgoTriMeshGeoStructHandle triMeshGeoHandle, const char *surfMeshFlattenedFile)
{
  pgo::Mesh::TriMeshGeo *triMeshGeoPtr = reinterpret_cast<pgo::Mesh::TriMeshGeo *>(triMeshGeoHandle);
  pgo::Mesh::TriMeshGeo &triMeshGeo = *triMeshGeoPtr;
  int n = triMeshGeo.numVertices();
  // scale the triMeshGeo
  for (int vi = 0; vi < n; vi++) {
    triMeshGeo.pos(vi) *= 10;
  }

  double minZ = 1e10;
  for (int vi = 0; vi < n; vi++) {
    minZ = std::min(minZ, triMeshGeo.pos(vi)[2]);
  }

  minZ += 5e-3;

  // create bbox with minZ
  pgo::Mesh::BoundingBox bbox(triMeshGeo.positions());
  // replace z with minZ
  std::vector<pgo::Vec3i> bboxMeshTri = {
    // Front face
    pgo::Vec3i(0, 2, 1),
    pgo::Vec3i(1, 2, 3),
    // Back face
    pgo::Vec3i(5, 7, 4),
    pgo::Vec3i(4, 7, 6),
    // Left face
    pgo::Vec3i(4, 6, 0),
    pgo::Vec3i(0, 6, 2),
    // Right face
    pgo::Vec3i(1, 3, 5),
    pgo::Vec3i(5, 3, 7),
    // Top face
    pgo::Vec3i(2, 6, 3),
    pgo::Vec3i(3, 6, 7),
    // Bottom face
    pgo::Vec3i(0, 1, 4),
    pgo::Vec3i(1, 5, 4)
  };
  std::vector<pgo::Vec3d> bboxMeshPos = {
    pgo::Vec3d(bbox.bmin()[0] - 1e-3, bbox.bmin()[1] - 1e-3, minZ),
    pgo::Vec3d(bbox.bmax()[0] + 1e-3, bbox.bmin()[1] - 1e-3, minZ),
    pgo::Vec3d(bbox.bmin()[0] - 1e-3, bbox.bmax()[1] + 1e-3, minZ),
    pgo::Vec3d(bbox.bmax()[0] + 1e-3, bbox.bmax()[1] + 1e-3, minZ),
    pgo::Vec3d(bbox.bmin()[0] - 1e-3, bbox.bmin()[1] - 1e-3, bbox.bmax()[2] + 1e-3),
    pgo::Vec3d(bbox.bmax()[0] + 1e-3, bbox.bmin()[1] - 1e-3, bbox.bmax()[2] + 1e-3),
    pgo::Vec3d(bbox.bmin()[0] - 1e-3, bbox.bmax()[1] + 1e-3, bbox.bmax()[2] + 1e-3),
    pgo::Vec3d(bbox.bmax()[0] + 1e-3, bbox.bmax()[1] + 1e-3, bbox.bmax()[2] + 1e-3)
  };
  pgo::Mesh::TriMeshGeo bboxMesh(bboxMeshPos, bboxMeshTri);

  // intersection
  pgo::Mesh::TriMeshGeo intersectMesh;
  bool isValidIntersect = pgo::CGALInterface::corefineAndComputeIntersection(triMeshGeo, bboxMesh, intersectMesh);
  if (!isValidIntersect) {
    std::cerr << "No intersection found\n";
    exit(1);
  }

  // scale back
  for (int v = 0; v < intersectMesh.numVertices(); v++) {
    intersectMesh.pos(v) /= 10;
  }

  intersectMesh.save(surfMeshFlattenedFile);
  return minZ;
}

void physcomp_stability_opt(const char *tetMeshFile, const char *fixedVtxFile, const char *saveFolder, int verbose)
{
  /////
  pgoTetMeshStructHandle tetmesh = pgo_create_tetmesh_from_file(tetMeshFile);

  int n = pgo_tetmesh_get_num_vertices(tetmesh);
  int nele = pgo_tetmesh_get_num_tets(tetmesh);

  std::cout << "n: " << n << " nele: " << nele << std::endl;

  std::vector<int> fixedVertices;

  std::ifstream ifs(fixedVtxFile);
  int v;
  while (ifs >> v) {
    fixedVertices.push_back(v);
  }

  const std::string saveFolderStr(saveFolder);

  pgoTetMeshStructHandle tetmeshRest = pgo_create_tetmesh_from_file(tetMeshFile);  // tetMeshStaticEqRes;  //
  pgoTetMeshStructHandle tetmeshInit = pgo_create_tetmesh_from_file(tetMeshFile);  // tetMeshStaticEqRes;  //

  std::vector<double> xOpt(n * 3 + nele * 6);
  pgoQuasiStaticSimStructHandle quasiStaticStruct = physcomp_create_quastic_static_sim_create_energies(tetmeshRest, tetmeshInit, fixedVertices.data(), (int)fixedVertices.size(), xOpt.data());

  // stand energy
  pgoCenterOfMassEnergyStructHandle standEnergy = physcomp_create_center_of_mass_energy(tetmeshInit);

  pgoTetMeshStructHandle tetmeshTarget = pgo_create_tetmesh_from_file(tetMeshFile);

  std::vector<double> xHat(n * 3);
  pgo_tetmesh_get_vertices(tetmeshTarget, xHat.data());

  // R(x) = ||x - x_hat||^2 * 0.5 + standEnergy * w
  // the mass center should be lying inside the support polygon
  // dR_dx = (x - x_hat) + d(standEnergy)/dx * w
  std::vector<double> dR_dx(n * 3);
  double stepSize = 0.2;
  // double stepSize = 1.0;
  double coeff = 10.0;

  std::vector<double> finalVtx(n * 3);
  std::vector<double> finalPlasticity(nele * 6);
  std::vector<double> xLastIter = xOpt;
  std::vector<double> CoM(3);

  std::vector<double> tgtCoM(3);
  std::vector<double> projMat(9);
  physcomp_center_of_mass_energy_get_tgtCoM_projMat(standEnergy, tgtCoM.data(), projMat.data());

  for (int iter = 0; iter < 1000; iter++) {
    double icpEng = 0.0;
    // if (iter < 15) {
    //   for (int i = 0; i < n * 3; i++) {
    //     dR_dx[i] = (xOpt[i] - xHat[i]);
    //     icpEng += dR_dx[i] * dR_dx[i] * 0.5;
    //   }
    // } else{
    //   for (int i = 0; i < n * 3; i++) {
    //     dR_dx[i] = (xOpt[i] - xHat[i]);
    //     icpEng += dR_dx[i] * dR_dx[i] * 0.5;
    //   }
    //     //   for (int i = 0; i < n * 3; i++) {
    //     //     dR_dx[i] = 0.0;
    //     //   }
    //   coeff = 1000;
    // }
    for (int i = 0; i < n * 3; i++) {
      dR_dx[i] = 0.0;
    }
    coeff = 1000;

    std::vector<double> standGrad(n * 3);
    double standEngVal = physcomp_center_of_mass_energy_gradient(standEnergy, xOpt.data(), standGrad.data(), CoM.data(), n);

    for (int i = 0; i < n * 3; i++) {
      dR_dx[i] += standGrad[i] * coeff;
    }

    std::vector<double> GNGrad(nele * 6);
    physcomp_quastic_static_gauss_newton_gradient(quasiStaticStruct, dR_dx.data(), xOpt.data(), n, nele, GNGrad.data(), true);

    for (int i = 0; i < nele * 6; i++) {
      xOpt[n * 3 + i] -= stepSize * GNGrad[i];
    }

    physcomp_project_plastic_param(quasiStaticStruct, n, nele, xOpt.data(), 1e-5);

    //
    physcomp_static_eq_step(quasiStaticStruct, xOpt.data(), n, nele, xOpt.data(), verbose);

    std::cout << "======iter: " << iter << " icp eng: " << icpEng << " standEngVal: " << standEngVal << std::endl;
    std::cout << "CoM: " << CoM[0] << " " << CoM[1] << " " << CoM[2] << std::endl;
    std::cout << "tgtCoM: " << tgtCoM[0] << " " << tgtCoM[1] << " " << tgtCoM[2] << std::endl;
    // break condition
    if (iter > 0) {
      double diff_max = 0.0;
      for (int i = 0; i < n * 3; i++) {
        diff_max = std::max(diff_max, std::abs(xOpt[i] - xLastIter[i]));
        xLastIter[i] = xOpt[i];
      }

      if (diff_max < 1e-9) {
        break;
      }
    }
  }

  for (int i = 0; i < n * 3; i++) {
    finalVtx[i] = xOpt[i];
  }

  for (int i = 0; i < nele * 6; i++) {
    finalPlasticity[i] = xOpt[n * 3 + i];
  }

  pgoTetMeshStructHandle tetMeshOpt = pgo_tetmesh_update_vertices(tetmeshInit, finalVtx.data());
  pgo_save_tetmesh_to_file(tetMeshOpt, (saveFolderStr + "/stand_opt.veg").c_str());

  std::vector<double> xRest(n * 3 + nele * 6);
  // use tetmesh as initial mesh
  physcomp_create_quastic_static_sim(tetmeshInit, fixedVertices.data(), (int)fixedVertices.size(), xRest.data(), finalPlasticity.data(), false);

  pgoTetMeshStructHandle tetMeshRestOpt = pgo_tetmesh_update_vertices(tetMeshOpt, xRest.data());
  pgo_save_tetmesh_to_file(tetMeshRestOpt, (saveFolderStr + "/stand_opt_rest.veg").c_str());
}
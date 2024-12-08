#pragma once

typedef struct pgoQuasiStaticSimStruct *pgoQuasiStaticSimStructHandle;
typedef struct pgoCenterOfMassEnergyStruct *pgoCenterOfMassEnergyStructHandle;

//// NeurIPS
void pgo_create_quastic_static_sim(pgoTetMeshStructHandle tetmeshHandle, int *fixedVerticesIDs, int numFixedVertices, double *xOpt, double *plasticityParam, bool with_gravity);
pgoQuasiStaticSimStructHandle pgo_create_quastic_static_sim_create_energies(pgoTetMeshStructHandle tetmeshHandle, pgoTetMeshStructHandle tetmeshInitHandle, int *fixedVerticesIDs, int numFixedVertices, double *xOpt, bool withGravity = true);
void pgo_quastic_static_gauss_newton_gradient(pgoQuasiStaticSimStructHandle quasiStaticStruct, double *dR_dx_data, double *var, int nVtx, int nElem, double *GNGrad, bool hasSmoothness = false);
void pgo_project_plastic_param(pgoQuasiStaticSimStructHandle quasiStaticStruct, int nVtx, int nElem, double *xOpt, double zeroThreshold);
void pgo_static_eq_step(pgoQuasiStaticSimStructHandle quasiStaticStruct, double *var, int nVtx, int nElem, double *varOut, int verbose = 3);
pgoCenterOfMassEnergyStructHandle pgo_create_center_of_mass_energy(pgoTetMeshStructHandle tetmeshHandle);
void pgo_center_of_mass_energy_get_tgtCoM_projMat(pgoCenterOfMassEnergyStructHandle comEnergyHandle, double *tgtCoM, double *projMat);
double pgo_center_of_mass_energy_gradient(pgoCenterOfMassEnergyStructHandle comEnergyHandle, const double *x, double *grad, double *com, int numVtx);

// our optimization
void pgo_inverse_plasticity_opt(const char *tetMeshFile, const char *fixedVtxFile, const char *saveFolder, double stepSize = 0.5, int verbose = 3);
double pgo_stablity_preprocess(pgoTriMeshGeoStructHandle triMeshGeo, const char *surfMeshFlattenedFile);
void pgo_stability_opt(const char *tetMeshFile, const char *fixedVtxFile, const char *saveFolder, int verbose = 3);
